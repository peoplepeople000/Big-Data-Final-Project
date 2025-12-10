import time
import json
from pathlib import Path
from domain import Domain
from tqdm.auto import tqdm  # auto detects notebook vs terminal
from requests.exceptions import Timeout as RequestsTimeout

class All_Domain:
    def __init__(self, parent_dir="all_city_data"):
        self.parent_dir = Path(parent_dir)
        self.parent_dir.mkdir(exist_ok=True)
        self.all_domain = []
        with open("socrata_domains_cities_only.txt") as f:
            for line in f:
                self.all_domain.append(Domain(line.strip()))
        self.base = self.parent_dir / "all_cities"
        self.base.mkdir(exist_ok=True)
    
    def all_counts(self, max_retries=3, initial_delay=2, max_timeout=60):
        """Get dataset counts for all domains with retry logic and progressive timeout."""
        results_file = self.base / "domain_counts.json"
        
        # Load existing results if available
        existing_results = {}
        if results_file.exists():
            print(f"Loading existing results from {results_file}")
            with open(results_file) as f:
                existing_data = json.load(f)
                existing_results = {r['domain']: r for r in existing_data}
        
        results = []
        
        # Progress bar for all domains
        pbar = tqdm(self.all_domain, desc="Fetching counts", unit="domain")
        
        for domain in pbar:
            domain_name = domain.domain
            pbar.set_description(f"Processing {domain_name[:30]}")
            
            # Use cached result if available and successful
            if domain_name in existing_results and existing_results[domain_name].get('count', -1) >= 0:
                count = existing_results[domain_name]['count']
                results.append(existing_results[domain_name])
                pbar.set_postfix({"status": "cached", "count": count})
                continue
            
            # Try with retries and progressive timeout
            success = False
            current_timeout = domain.timeout
            
            for attempt in range(max_retries):
                try:
                    # Temporarily increase timeout for this attempt
                    domain.timeout = min(current_timeout * (attempt + 1), max_timeout)
                    
                    count = domain.city_datasets_count()
                    results.append({
                        "domain": domain_name,
                        "count": count
                    })
                    pbar.set_postfix({"status": "Success", "count": count})
                    success = True
                    break
                    
                except RequestsTimeout as e:
                    # Specific handling for requests timeout
                    if attempt < max_retries - 1:
                        new_timeout = min(domain.timeout * 1.5, max_timeout)
                        pbar.set_postfix({
                            "status": f"timeout, retry {attempt+1}/{max_retries}", 
                            "timeout": f"{new_timeout}s"
                        })
                        domain.timeout = new_timeout
                        time.sleep(initial_delay)
                    else:
                        pbar.set_postfix({"status": "Timeout", "tried": f"{domain.timeout}s"})
                        results.append({
                            "domain": domain_name,
                            "count": -1,
                            "error": f"Timeout after {domain.timeout}s"
                        })
                        
                except Exception as e:
                    # Other errors
                    if attempt < max_retries - 1:
                        pbar.set_postfix({"status": f"retry {attempt+1}/{max_retries}"})
                        time.sleep(initial_delay)
                    else:
                        pbar.set_postfix({"status": "Failed"})
                        results.append({
                            "domain": domain_name,
                            "count": -1,
                            "error": str(e)
                        })
            
            # Reset timeout for next domain
            domain.timeout = current_timeout
            
            # Save after each domain (incremental saves)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        pbar.close()
        
        # Print summary
        successful = [r for r in results if r['count'] >= 0]
        failed_timeout = [r for r in results if 'Timeout' in r.get('error', '')]
        total_datasets = sum(r['count'] for r in successful)
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"Successful: {len(successful)}/{len(results)} domains")
        print(f"Failed: {len(results) - len(successful)} domains")
        print(f"(Timeouts: {len(failed_timeout)})")
        print(f"Total datasets: {total_datasets:,}")
        print(f"Results saved to {results_file}")
        print(f"{'='*60}")
        
        return results

    def download_all_metadata(self, resume=True, max_retries=3, max_timeout=120):
        """Download metadata for all domains with progress tracking."""
        progress_file = self.base / "download_progress.json"
        
        # Load progress if resuming
        completed = set()
        if resume and progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
                completed = set(progress.get('completed', []))
                print(f"Resuming... Already completed: {len(completed)} domains")
        
        domains_to_process = [d for d in self.all_domain if d.domain not in completed]
        print(f"Processing {len(domains_to_process)} domains...")
        
        # Progress bar
        pbar = tqdm(domains_to_process, desc="Downloading metadata", unit="domain")
        results = []
        
        for domain in pbar:
            domain_name = domain.domain
            pbar.set_description(f"Downloading {domain_name[:30]}")
            
            # Domain handles all retries and timeouts internally
            result = domain.download_all_relevant_metadata(
                max_retries=max_retries,
                progressive_timeout=True,
                max_timeout=max_timeout
            )
            
            results.append({'domain': domain_name, **result})
            
            # Mark as completed if any datasets succeeded
            if result['successful'] > 0:
                completed.add(domain_name)
            
            pbar.set_postfix({
                "success": result['successful'],
                "failed": len(result['failed'])
            })
            
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump({
                    'completed': list(completed),
                    'total': len(self.all_domain),
                    'last_updated': time.time()
                }, f, indent=2)
        
        pbar.close()
        
        # Summary
        total_successful = sum(r['successful'] for r in results)
        total_failed = sum(len(r['failed']) for r in results)
        
        print(f"\n{'='*60}")
        print(f"Metadata download complete!")
        print(f"Successful downloads: {total_successful:,}")
        print(f"Failed downloads: {total_failed:,}")
        print(f"Domains processed: {len(completed)}/{len(self.all_domain)}")
        print(f"{'='*60}")
        
        return results
    
    def cleanup_empty_metadata(self, dry_run=True, update_progress=True):
        """
        Find and optionally delete empty or null JSON files in metadata directories.
        Also updates progress file to mark domains with empty files as incomplete.
        
        Args:
            dry_run: If True, only report what would be deleted without actually deleting
            update_progress: If True, remove domains with empty files from completed list
        
        Returns:
            Dictionary with cleanup statistics
        """
        empty_files = []
        corrupted_files = []
        domains_with_issues = set()
        
        print(f"Scanning metadata directories for empty/null files...")
        
        pbar = tqdm(self.all_domain, desc="Scanning domains", unit="domain")
        
        for domain in pbar:
            pbar.set_description(f"Scanning {domain.domain[:30]}")
            
            if not domain.metadatadir.exists():
                continue
            
            json_files = list(domain.metadatadir.glob("metadata_*.json"))
            domain_has_issues = False
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        content = json.load(f)
                        
                    # Check if empty/null
                    if content is None or (isinstance(content, dict) and not content):
                        empty_files.append(json_file)
                        domain_has_issues = True
                        
                except json.JSONDecodeError:
                    # File is corrupted
                    corrupted_files.append(json_file)
                    domain_has_issues = True
                except Exception as e:
                    print(f"\n  Warning: Could not read {json_file}: {e}")
                    domain_has_issues = True
            
            if domain_has_issues:
                domains_with_issues.add(domain.domain)
        
        pbar.close()
        
        # Report findings
        print(f"\n{'='*60}")
        print(f"Cleanup scan complete:")
        print(f"Empty/null files: {len(empty_files)}")
        print(f"Corrupted files: {len(corrupted_files)}")
        print(f"Total files to remove: {len(empty_files) + len(corrupted_files)}")
        print(f"Domains affected: {len(domains_with_issues)}")
        
        # Show which domains have issues
        if domains_with_issues:
            print(f"\n  Domains with empty/corrupted files:")
            for domain_name in sorted(list(domains_with_issues))[:10]:
                print(f"- {domain_name}")
            if len(domains_with_issues) > 10:
                print(f"... and {len(domains_with_issues) - 10} more")
        
        if empty_files or corrupted_files:
            if dry_run:
                print(f"\nThis is a DRY RUN - no files deleted, progress not updated.")
                print(f"Run with dry_run=False to actually delete files and update progress.")
                
                # Show some examples
                if empty_files:
                    print(f"\n  Example empty files:")
                    for f in empty_files[:5]:
                        print(f"- {f}")
                    if len(empty_files) > 5:
                        print(f"... and {len(empty_files) - 5} more")
                
                if corrupted_files:
                    print(f"\nExample corrupted files:")
                    for f in corrupted_files[:5]:
                        print(f"- {f}")
                    if len(corrupted_files) > 5:
                        print(f"... and {len(corrupted_files) - 5} more")
            else:
                # Actually delete files
                print(f"\n  Deleting files...")
                deleted = 0
                failed_to_delete = []
                
                all_to_delete = empty_files + corrupted_files
                for file_path in tqdm(all_to_delete, desc="Deleting", unit="file"):
                    try:
                        file_path.unlink()
                        deleted += 1
                    except Exception as e:
                        failed_to_delete.append((file_path, str(e)))
                
                print(f"\n  Deleted: {deleted} files")
                if failed_to_delete:
                    print(f"  Failed to delete: {len(failed_to_delete)} files")
                    for path, error in failed_to_delete[:5]:
                        print(f"    - {path}: {error}")
                
                # Update progress file if requested
                if update_progress and domains_with_issues:
                    progress_file = self.base / "download_progress.json"
                    
                    if progress_file.exists():
                        print(f"\n  Updating progress file...")
                        
                        with open(progress_file, 'r') as f:
                            progress = json.load(f)
                        
                        original_completed = set(progress.get('completed', []))
                        updated_completed = original_completed - domains_with_issues
                        
                        progress['completed'] = list(updated_completed)
                        progress['last_updated'] = time.time()
                        
                        with open(progress_file, 'w') as f:
                            json.dump(progress, f, indent=2)
                        
                        removed_count = len(original_completed) - len(updated_completed)
                        print(f"Removed {removed_count} domains from completed list")
                        print(f"These domains will be reprocessed on next download_all_metadata(resume=True)")
                    else:
                        print(f"\nNo progress file found at {progress_file}")
                        print(f"Progress tracking will start fresh on next download.")
        else:
            print(f"\n  No empty or corrupted files found!")
        
        print(f"{'='*60}")
        
        return {
            'empty_files': empty_files,
            'corrupted_files': corrupted_files,
            'domains_affected': list(domains_with_issues),
            'total': len(empty_files) + len(corrupted_files)
        }

    def redownload_missing_metadata(self, max_retries=3, max_timeout=120):
        """
        Find domains with missing metadata and re-download.
        Useful after cleanup_empty_metadata or failed downloads.
        """
        print("Checking for domains with missing metadata...")
        
        domains_to_redownload = []
        failed_to_check = []
        
        # Check each domain
        pbar_check = tqdm(self.all_domain, desc="Checking domains", unit="domain")
        
        for domain in pbar_check:
            pbar_check.set_description(f"Checking {domain.domain[:30]}")
            
            # Get expected number of datasets with retry logic
            expected_ids = None
            original_timeout = domain.timeout
            
            for attempt in range(max_retries):
                try:
                    # Increase timeout progressively
                    domain.timeout = min(original_timeout * (attempt + 1), max_timeout)
                    
                    # Try to get IDs (this might timeout)
                    expected_ids = set(domain._set_up_ids())
                    
                    # Verify we got something
                    if not expected_ids:
                        raise ValueError("No dataset IDs returned")
                    
                    break  # Success!
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_timeout = 'timeout' in error_str or 'timed out' in error_str
                    
                    if attempt < max_retries - 1:
                        if is_timeout:
                            new_timeout = min(domain.timeout * 1.5, max_timeout)
                            pbar_check.set_postfix({
                                "status": f"timeout, retry {attempt+1}/{max_retries}",
                                "timeout": f"{new_timeout}s"
                            })
                        else:
                            pbar_check.set_postfix({"status": f"retry {attempt+1}/{max_retries}"})
                        
                        time.sleep(2)  # Brief pause before retry
                        
                    else:
                        # All retries failed
                        pbar_check.set_postfix({"status": "Failed to get IDs"})
                        tqdm.write(f"Failed {domain.domain}: Could not fetch dataset IDs after {max_retries} attempts - {e}")
                        expected_ids = None
                        failed_to_check.append(domain.domain)
                        break
            
            # Reset timeout
            domain.timeout = original_timeout
            
            # Skip if we couldn't get IDs
            if expected_ids is None or not expected_ids:
                pbar_check.set_postfix({"status": "skipped"})
                continue
            
            # Now safe to use expected_ids
            # Get actual metadata files
            existing_ids = set()
            
            if domain.metadatadir.exists():
                existing_files = list(domain.metadatadir.glob("metadata_*.json"))
                
                for f in existing_files:
                    # Extract ID from filename: metadata_<id>.json
                    dataset_id = f.stem.replace("metadata_", "")
                    existing_ids.add(dataset_id)
                
                missing = expected_ids - existing_ids
                
                if missing:
                    domains_to_redownload.append({
                        'domain': domain,
                        'missing_count': len(missing),
                        'total_count': len(expected_ids)
                    })
            else:
                # No metadata directory at all
                domains_to_redownload.append({
                    'domain': domain,
                    'missing_count': len(expected_ids),
                    'total_count': len(expected_ids)
                })
        
        pbar_check.close()
        
        # Report checking results
        if failed_to_check:
            print(f"\nWarning: Could not check {len(failed_to_check)} domains:")
            for domain_name in failed_to_check[:5]:
                print(f"- {domain_name}")
            if len(failed_to_check) > 5:
                print(f"... and {len(failed_to_check) - 5} more")
            print(f"  These domains may have missing metadata but couldn't be verified.")
        
        if not domains_to_redownload:
            print("\n  All checked domains have complete metadata!")
            return []
        
        print(f"\n{'='*60}")
        print(f"Found {len(domains_to_redownload)} domains with missing metadata:")
        for info in domains_to_redownload[:10]:
            print(f"  - {info['domain'].domain}: {info['missing_count']}/{info['total_count']} missing")
        if len(domains_to_redownload) > 10:
            print(f"  ... and {len(domains_to_redownload) - 10} more")
        print(f"{'='*60}")
        
        # Ask for confirmation
        response = input(f"\nRedownload metadata for these {len(domains_to_redownload)} domains? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return []
        
        # Re-download
        print("\nRedownloading...")
        results = []
        
        pbar = tqdm(domains_to_redownload, desc="Redownloading", unit="domain")
        
        for info in pbar:
            domain = info['domain']
            pbar.set_description(f"Redownloading {domain.domain[:30]}")
            
            result = domain.download_all_relevant_metadata(
                max_retries=max_retries,
                progressive_timeout=True,
                max_timeout=max_timeout
            )
            
            results.append({
                'domain': domain.domain,
                **result
            })
            
            pbar.set_postfix({
                "success": result['successful'],
                "failed": len(result['failed'])
            })
        
        pbar.close()
        
        # Summary
        total_successful = sum(r['successful'] for r in results)
        total_failed = sum(len(r['failed']) for r in results)
        
        print(f"\n{'='*60}")
        print(f"Redownload complete:")
        print(f"Successful: {total_successful}")
        print(f"Failed: {total_failed}")
        if failed_to_check:
            print(f"Could not verify: {len(failed_to_check)} domains")
        print(f"{'='*60}")
        
        return results
    
    def retry_failed_counts(self, max_retries=3, initial_delay=2, max_timeout=60):
        """Retry only the domains that failed in all_counts() with progressive timeout."""
        results_file = self.base / "domain_counts.json"
        
        if not results_file.exists():
            print("No existing results found. Run all_counts() first.")
            return
        
        with open(results_file) as f:
            existing_results = json.load(f)
        
        # Find failed domains
        failed_domains = [r for r in existing_results if r['count'] < 0]
        
        if not failed_domains:
            print("No failed domains to retry!")
            return
        
        print(f"Retrying {len(failed_domains)} failed domains...")
        
        # Create a lookup dict
        results_dict = {r['domain']: r for r in existing_results}
        
        # Progress bar for retries
        pbar = tqdm(failed_domains, desc="Retrying failed", unit="domain")
        
        for failed in pbar:
            domain_name = failed['domain']
            pbar.set_description(f"Retrying {domain_name[:30]}")
            
            domain = next((d for d in self.all_domain if d.domain == domain_name), None)
            
            if not domain:
                pbar.set_postfix({"status": "not found"})
                continue
            
            # Try with retries and progressive timeout
            success = False
            current_timeout = domain.timeout
            
            for attempt in range(max_retries):
                try:
                    # Increase timeout progressively
                    domain.timeout = min(current_timeout * (attempt + 1), max_timeout)
                    
                    count = domain.city_datasets_count()
                    results_dict[domain_name] = {
                        "domain": domain_name,
                        "count": count
                    }
                    pbar.set_postfix({"status": "Success", "count": count})
                    success = True
                    break
                    
                except RequestsTimeout as e:
                    # Specific handling for requests timeout
                    if attempt < max_retries - 1:
                        new_timeout = min(domain.timeout * 1.5, max_timeout)
                        pbar.set_postfix({
                            "status": f"timeout, retry {attempt+1}/{max_retries}",
                            "timeout": f"{new_timeout}s"
                        })
                        domain.timeout = new_timeout
                        time.sleep(initial_delay)
                    else:
                        pbar.set_postfix({"status": "Still timeout"})
                        results_dict[domain_name] = {
                            "domain": domain_name,
                            "count": -1,
                            "error": f"Timeout after {domain.timeout}s"
                        }
                        
                except Exception as e:
                    # Other errors
                    if attempt < max_retries - 1:
                        pbar.set_postfix({"status": f"retry {attempt+1}/{max_retries}"})
                        time.sleep(initial_delay)
                    else:
                        pbar.set_postfix({"status": "Still failing"})
                        tqdm.write(f"{domain_name}: {e}")
                        results_dict[domain_name] = {
                            "domain": domain_name,
                            "count": -1,
                            "error": str(e)
                        }
            
            # Reset timeout for next domain
            domain.timeout = current_timeout
        
        pbar.close()
        
        # Save updated results
        updated_results = list(results_dict.values())
        with open(results_file, 'w') as f:
            json.dump(updated_results, f, indent=2)
        
        successful = [r for r in updated_results if r['count'] >= 0]
        failed_timeout = [r for r in updated_results if r['count'] < 0 and 'Timeout' in r.get('error', '')]
        
        print(f"\n{'='*60}")
        print(f"Retry complete:")
        print(f"Now successful: {len(successful)}/{len(updated_results)} domains")
        print(f"Still failed: {len(updated_results) - len(successful)} domains")
        print(f"(Still timing out: {len(failed_timeout)})")
        print(f"{'='*60}")
        
        return updated_results
    
    def all_metadata_summaries(self, skip_missing=True):
        """Generate metadata summaries for all domains that have them."""
        results = []
        
        # Progress bar
        pbar = tqdm(self.all_domain, desc="Generating summaries", unit="domain")
        
        for domain in pbar:
            domain_name = domain.domain
            pbar.set_description(f"Summarizing {domain_name[:30]}")
            
            # Check if metadata exists before trying to summarize
            json_files = list(domain.metadatadir.glob("*.json"))
            if not json_files:
                if skip_missing:
                    pbar.set_postfix({"status": "skipped", "reason": "no metadata"})
                    results.append({
                        "domain": domain_name,
                        "status": "skipped",
                        "reason": "no_metadata"
                    })
                    continue
            
            try:
                domain.summarize_metadata()
                pbar.set_postfix({"status": "Sucess", "files": len(json_files)})
                results.append({
                    "domain": domain_name,
                    "status": "success",
                    "files_processed": len(json_files)
                })
            except Exception as e:
                pbar.set_postfix({"status": "Failed"})
                tqdm.write(f"  ✗ {domain_name}: {e}")
                results.append({
                    "domain": domain_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        pbar.close()
        
        # Save results log
        output_file = self.base / "summary_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        successful = [r for r in results if r['status'] == 'success']
        skipped = [r for r in results if r['status'] == 'skipped']
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Successful: {len(successful)} domains")
        print(f"  Skipped: {len(skipped)} domains")
        print(f"  Failed: {len(results) - len(successful) - len(skipped)} domains")
        print(f"  Results log saved to {output_file}")
        print(f"{'='*60}")
        
        return results
    
    def aggregate_summaries(self):
        """Aggregate all individual domain summaries into overall statistics."""
        from collections import Counter
        
        # Use Counters for easy aggregation
        tag_counter = Counter()
        category_counter = Counter()
        format_counter = Counter()
        column_type_counter = Counter()
        publication_counter = Counter()
        update_counter = Counter()
        attribute_counter = Counter()
        view_counter = Counter()
        download_counter = Counter()
        row_counter = Counter()
        sparseness_counter = Counter()
        
        # Progress bar for aggregation
        pbar = tqdm(self.all_domain, desc="Aggregating summaries", unit="domain")
        
        for domain in pbar:
            summary_dir = domain.metadatadir / "summary"
            pbar.set_description(f"Aggregating {domain.domain[:30]}")
            
            if not summary_dir.exists():
                pbar.set_postfix({"status": "no summary"})
                continue
            
            pbar.set_postfix({"status": "Success"})
            
            # Aggregate tag counts
            tag_file = summary_dir / "tag_counts.json"
            if tag_file.exists():
                with open(tag_file) as f:
                    tags = json.load(f)
                    # Now it's a dict like {"Balloon": 13, "Crime": 25}
                    tag_counter.update(tags)
            
            # Aggregate publication age counts
            publication_file = summary_dir / "publication_age.json"
            if publication_file.exists():
                with open(publication_file) as f:
                    publications = json.load(f)
                    # Convert string keys to int if they're age_months
                    publication_counter.update({int(k): v for k, v in publications.items()})

            # Aggregate update age counts
            update_file = summary_dir / "last_update.json"
            if update_file.exists():
                with open(update_file) as f:
                    updates = json.load(f)
                    # Convert string keys to int if they're age_months
                    update_counter.update({int(k): v for k, v in updates.items()})
            
            # Aggregate categories
            cat_file = summary_dir / "categories.json"
            if cat_file.exists():
                with open(cat_file) as f:
                    cats = json.load(f)
                    category_counter.update(cats)
            
            # Aggregate formats
            fmt_file = summary_dir / "formats.json"
            if fmt_file.exists():
                with open(fmt_file) as f:
                    fmts = json.load(f)
                    format_counter.update(fmts)
            
            # Aggregate column types
            type_file = summary_dir / "column_types.json"
            if type_file.exists():
                with open(type_file) as f:
                    types = json.load(f)
                    column_type_counter.update(types)
            
            # Aggregate attribute buckets
            attribute_file = summary_dir / "attribute_counts.json"
            if attribute_file.exists():
                with open(attribute_file) as f:
                    attributes = json.load(f)
                    attribute_counter.update(attributes)

            # Aggregate view buckets
            view_file = summary_dir / "view_buckets.json"
            if view_file.exists():
                with open(view_file) as f:
                    views = json.load(f)
                    view_counter.update(views)
            
            # Aggregate download buckets
            download_file = summary_dir / "download_buckets.json"
            if download_file.exists():
                with open(download_file) as f:
                    downloads = json.load(f)
                    download_counter.update(downloads)

            # Aggregate row buckets
            row_file = summary_dir / "row_buckets.json"
            if row_file.exists():
                with open(row_file) as f:
                    rows = json.load(f)
                    row_counter.update(rows)

            # Aggregate sparseness buckets
            sparseness_file = summary_dir / "table_sparseness.json"
            if sparseness_file.exists():
                with open(sparseness_file) as f:
                    sparsenesss = json.load(f)
                    sparseness_counter.update(sparsenesss)

        pbar.close()

        # Define bucket orders
        BUCKET_ORDERS = {
            'attribute': ['0-10', '10-20', '20-30', '30-40', '40-50', '50+'],
            'download': ['0-100', '100-1K', '1K-10K', '10K+'],
            'row': ['0-1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M+'],
            'view': ['0-100', '100-1K', '1K-10K', '10K+'],
            'sparseness': ['< 1% sparse', '1-5% sparse', '5-10% sparse', '10-25% sparse', '25-50% sparse', '50%+ sparse']
        }

        def sort_by_bucket_order(counter_dict, order_list):
            """Sort a counter dictionary by predefined bucket order."""
            return {k: counter_dict[k] for k in order_list if k in counter_dict}

        # Convert Counters to the new format (simple dicts)
        output = {
            "tag_counts": dict(tag_counter.most_common()),
            "publication_age": dict(sorted(publication_counter.items(), key=lambda x: x[0])),
            "last_update": dict(sorted(update_counter.items(), key=lambda x: x[0])),
            "categories": dict(category_counter.most_common()),
            "formats": dict(format_counter.most_common()),
            "column_types": dict(column_type_counter.most_common()),
            "attribute_counts": sort_by_bucket_order(attribute_counter, BUCKET_ORDERS['attribute']),
            "download_counts": sort_by_bucket_order(download_counter, BUCKET_ORDERS['download']),
            "row_counts": sort_by_bucket_order(row_counter, BUCKET_ORDERS['row']),
            "view_counts": sort_by_bucket_order(view_counter, BUCKET_ORDERS['view']),
            "sparseness_counts": sort_by_bucket_order(sparseness_counter, BUCKET_ORDERS['sparseness']),
        }
        
        print("\nSaving aggregated results...")
        # Save aggregated results
        for name, data in output.items():
            output_file = self.base / f"aggregated_{name}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f" Saved {output_file}")
        
        print(f"\n{'='*60}")
        print("Aggregation complete!")
        print(f"{'='*60}")
        
        return output
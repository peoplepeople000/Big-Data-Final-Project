import math
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from adjustText import adjust_text
from tqdm.auto import tqdm
from pathlib import Path

class All_Domain_Plotter:
    def __init__(self, all_domain):
        self.all_domain = all_domain
        self.plots = self.all_domain.base / "plots"

    def tagcloud(self, domain = None, log = True, w=800, h=400, max_words=100, save_name=None):
        freqs = self.all_domain.base / f"aggregated_tag_counts.json"
        if domain:
            if domain in [d.domain for d in self.all_domain.all_domain]:
                freqs = self.all_domain.parent_dir / domain / "metadata" / "summary" / "tag_counts.json"
            else: return None
        if not (freqs).exists():
            return None
        with (freqs).open() as f:
            wc = WordCloud(width=w, height=h, max_words=max_words, background_color="white")
            counts = json.load(f)
            if log:
                counts = {tag: math.log(count + 1) for tag, count in counts.items()}
            SKIPPED_TAGS = ["tif", "kml", "kmz", "gis"]
            filtered = {tag: count for tag, count in counts.items() if not any(bad in tag.lower() for bad in SKIPPED_TAGS)}
            wc.generate_from_frequencies(filtered)
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            if save_name:
                self._ensure_plots()
                plt.savefig(self.plots / save_name, dpi=300, bbox_inches='tight')
                print(f"Tag cloud saved to {self.plots / save_name}")
            else:
                plt.show()

    def pop_count(self, save_name = None):
        counts_df = pd.DataFrame.from_dict(self.all_domain.all_counts())
        pops = self.all_domain.base / "city_pops.csv"
        pops_df = pd.read_csv(pops)
        merged_df = pd.merge(counts_df, pops_df, on='domain')
        # ----- pick extremes -----
        small_pop   = merged_df.nsmallest(3, "population")
        large_pop   = merged_df.nlargest(3, "population")
        small_count = merged_df.nsmallest(2, "count")
        large_count = merged_df.nlargest(2, "count")
        labeled_df = pd.concat([small_pop, large_pop, small_count, large_count]).drop_duplicates()
        # ----- plot -----
        ax = merged_df.plot(kind="scatter", x="population", y="count", logx=True, logy=True, figsize=(9,6))
        texts = []
        for _, row in labeled_df.iterrows():
            texts.append(
                ax.text(
                    row["population"],
                    row["count"],
                    row["city"],
                    fontsize=9,
                    alpha=0.9
                )
            )
        adjust_text(texts, ax=ax, only_move={"text": "y"})
        # --- add trendline ---
        x = merged_df["population"].values
        y = merged_df["count"].values
        logx = np.log10(x)
        logy = np.log10(y)
        b, a = np.polyfit(logx, logy, 1)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = 10**(a + b * np.log10(x_fit))
        ax.plot(x_fit, y_fit, linewidth=2, color="red", alpha = 0.5, linestyle='dotted')
        plt.title('Number of Datasets vs Population', fontsize=16)
        if save_name:
            self._ensure_plots()
            plt.savefig(self.plots / save_name, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to {self.plots / save_name}")
        else:
            plt.show()

    def plot_format_distribution(self, save_name=None):
        """
        Create a pie chart showing the distribution of dataset formats across all cities.
        
        Args:
            save_path: Optional path to save the figure. If None, displays interactively.
        """
        # Load aggregated format data
        formats_file = self.all_domain.base / "aggregated_formats.json"
        
        if not formats_file.exists():
            print("Aggregated formats not found. Run aggregate_summaries() first.")
            return
        
        with open(formats_file) as f:
            formats_data = json.load(f)
        
        if not formats_data:
            print("No format data available.")
            return
        
        # Prepare data for pie chart
        labels = list(formats_data.keys())
        sizes = list(formats_data.values())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Dataset Format Distribution Across All Cities', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Add legend with counts
        legend_labels = [f'{label}: {count:,}' for label, count in zip(labels, sizes)]
        ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        if save_name:
            self._ensure_plots()
            plt.savefig(self.plots / save_name, dpi=300, bbox_inches='tight')
            print(f"Pie chart saved to {self.plots / save_name}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_tabular_percentage_by_city(self, save_name=None, top_n=None, min_datasets=10):
        """
        Create a bar chart showing the percentage of tabular datasets for each city.
        
        Args:
            save_path: Optional path to save the figure. If None, displays interactively.
            top_n: If specified, only show top N cities by total dataset count
            min_datasets: Minimum number of datasets required to include a city
        """
        print("Calculating tabular percentages for each city...")
        
        city_stats = []
        
        for domain in tqdm(self.all_domain.all_domain, desc="Processing cities", unit="city"):
            summary_dir = domain.metadatadir / "summary"
            
            if not summary_dir.exists():
                continue
            
            # Load format data for this city
            formats_file = summary_dir / "formats.json"
            if not formats_file.exists():
                continue
            
            with open(formats_file) as f:
                formats = json.load(f)
            
            # Calculate total datasets and tabular count
            total_datasets = sum(formats.values())
            tabular_count = formats.get('tabular', 0)
            
            # Skip cities with too few datasets
            if total_datasets < min_datasets:
                continue
            
            tabular_percentage = (tabular_count / total_datasets * 100) if total_datasets > 0 else 0
            
            city_stats.append({
                'city': domain.domain,
                'total_datasets': total_datasets,
                'tabular_count': tabular_count,
                'tabular_percentage': tabular_percentage
            })
        
        if not city_stats:
            print("No city data available. Run summarize_metadata() for individual domains first.")
            return
        
        # Sort by tabular percentage (descending)
        city_stats.sort(key=lambda x: x['tabular_percentage'], reverse=True)
        
        # Limit to top N if specified
        if top_n:
            city_stats = city_stats[:top_n]
            title_suffix = f' (Top {top_n} Cities)'
        else:
            title_suffix = f' ({len(city_stats)} Cities)'
        
        # Prepare data for bar chart
        cities = [stat['city'] for stat in city_stats]
        percentages = [stat['tabular_percentage'] for stat in city_stats]
        totals = [stat['total_datasets'] for stat in city_stats]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(cities) * 0.3)))
        
        # Create horizontal bar chart
        bars = ax.barh(cities, percentages, color='steelblue')
        
        # Add percentage labels on bars (inside or outside depending on bar width)
        for i, (bar, pct, total) in enumerate(zip(bars, percentages, totals)):
            width = bar.get_width()
            label = f'{pct:.1f}% ({total:,} datasets)'
            
            # If bar is wide enough (>40%), put label inside on the left
            if width > 40:
                ax.text(
                    5,  # Small offset from left edge
                    bar.get_y() + bar.get_height()/2,
                    label,
                    ha='left', 
                    va='center',
                    fontsize=9,
                    color='white',
                    fontweight='bold'
                )
            else:
                # Otherwise put label outside on the right
                ax.text(
                    width + 1, 
                    bar.get_y() + bar.get_height()/2,
                    label,
                    ha='left', 
                    va='center',
                    fontsize=9
                )
        
        # Formatting
        ax.set_xlabel('Percentage of Tabular Datasets', fontsize=12, fontweight='bold')
        ax.set_ylabel('City Domain', fontsize=12, fontweight='bold')
        ax.set_title(f'Percentage of Tabular Datasets by City{title_suffix}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 105)  # Just enough space for outside labels
        
        # Add grid for easier reading
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_name:
            self._ensure_plots()
            plt.savefig(self.plots / save_name, dpi=300, bbox_inches='tight')
            print(f"Bar chart saved to {self.plots / save_name}")
        else:
            plt.show()
        
        plt.close()
            
    def plot_both_format_visualizations(self):
        """
        Generate both format visualizations and save them.
        
        Args:
            output_dir: Directory to save plots. If None, uses self.base / "plots"
        """
        self._ensure_plots()
        
        print("Generating format visualizations...")
        
        # Pie chart
        print("\n1. Creating format distribution pie chart...")
        self.plot_format_distribution(save_path=self.plots / "format_distribution_pie.png")
        
        # Bar chart - top 30 cities
        print("\n2. Creating tabular percentage bar chart (top 30 cities)...")
        self.plot_tabular_percentage_by_city(
            save_path=self.plots / "tabular_percentage_top30.png",
            top_n=30
        )
        
        # Bar chart - all cities
        print("\n3. Creating tabular percentage bar chart (all cities)...")
        self.plot_tabular_percentage_by_city(
            save_path=self.plots / "tabular_percentage_all.png"
        )
        
        print(f"\nAll visualizations saved to {self.plots}")

    def row_count(self):
        self._ensure_plots()
        row_file = self.all_domain.base / "aggregated_row_counts.json"
        with open(row_file) as f:
            rows = json.load(f)
            rows_df = pd.Series(rows).to_frame("Percentage of Total")
            new_rows = rows_df.rename_axis("No. of Records").reset_index()
            new_rows["Percentage of Total"] = (new_rows["Percentage of Total"] / new_rows["Percentage of Total"].sum() * 100).round(2)
            new_rows.to_html(self.plots / 'Row_Count_Data.html', index=False)
            print(f"Table saved to {self.plots / 'Row_Count_Data.html'}")

    def data_count_cat(self, top_n=3):
        """
        Get dataset counts and top categories for each city.
        
        Args:
            top_n: Number of top categories to include (default 3)
            exclude: List of categories to exclude (default: common junk categories)
        """        
        counts_df = pd.DataFrame.from_dict(self.all_domain.all_counts())
        pops = self.all_domain.base / "city_pops.csv"
        pops_df = pd.read_csv(pops)
        merged_df = pd.merge(counts_df, pops_df, on='domain')
        
        # Collect top categories
        category_data = []
        
        for domain in self.all_domain.all_domain:
            cat_file = domain.metadatadir / "summary" / "categories.json"
            
            if cat_file.exists():
                with open(cat_file) as f:
                    cats = json.load(f)                
                top_cats = [cat for cat, _ in list(cats.items())[:top_n]]
            else:
                top_cats = []

            if not top_cats:
                top_cats = pd.NA
            
            category_data.append({
                'domain': domain.domain,
                'top_categories': top_cats,
            })
        
        categories_df = pd.DataFrame(category_data)
        df = pd.merge(merged_df, categories_df, on='domain')
        df = df.sort_values(by="count", ascending=True)
        # Create a serial-number column (1-based)
        df["S. No."] = range(1, len(df) + 1)
        # Choose columns in desired order
        keep_columns = ["S. No.", "city", "count", "top_categories"]
        df = df[keep_columns]
        # Rename columns
        final_df = df.rename(columns={
            "city": "City Name",
            "count": "No. of Datasets",
            "top_categories": "Top-Three Categories"
        })

        self._ensure_plots()
        final_df.to_html(self.plots / 'City_Set_Count_And_Categories.html', index=False)
        print(f"Table saved to {self.plots / 'City_Set_Count_And_Categories.html'}")

    def bar_graph(self, data, save_name = None):
        ACT = {
            "attribute": "aggregated_attribute_counts.json",
            "download": "aggregated_download_counts.json", 
            "view": "aggregated_view_counts.json",
            "sparseness": "aggregated_sparseness_counts.json",
            "type": "aggregated_column_types.json"
            }
        file = ACT.get(data)
        if not file:
            print(f"{data} cannot be displayed with a bar graph.")
            return
        path = self.all_domain.base / file
        if path.exists():
            with open(path) as f:
                counts = json.load(f)    
        else:
            print(f"{path} not found") 
            return None              
        caps = data.capitalize()
        series = pd.Series(counts)
        ax = series.plot(kind="bar", figsize=(9,6))
        plt.xlabel(data.capitalize(), fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, v in enumerate(series.values):
            ax.text(i, v + max(series.values) * 0.01, f'{v:,}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.title(f'{caps} vs Count', fontsize=16)

        if save_name:
            self._ensure_plots()
            plt.savefig(self.plots / save_name, dpi=300, bbox_inches='tight')
            print(f"Bar chart for {data} saved to {self.plots / save_name}")
        else:
            plt.show()

    def publication_age(self, save_name = None):
        path = self.all_domain.base / "aggregated_publication_age.json"
        if path.exists():
            with open(path) as f:
                counts = json.load(f)
        else:
            print(f"{path} not found") 
            return None           
        # Prepare data
        series = pd.Series(counts)
        series.index = series.index.astype(int)
        series = series.sort_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot line
        ax.plot(series.index, series.values, linewidth=2.5, 
                marker='o', markersize=5, color='steelblue', label='Dataset Count')
        
        # Add reference lines
        ax.axvline(x=12, color='red', linestyle='--', alpha=0.5, linewidth=1, label='1 year')
        ax.axvline(x=24, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='2 years')
        ax.axvline(x=60, color='green', linestyle='--', alpha=0.5, linewidth=1, label='5 years')
        
        # Formatting
        ax.set_title('Dataset Age Distribution (All Cities)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Age in Months', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Datasets', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        
        # Add value labels for peaks
        max_idx = series.idxmax()
        max_val = series.max()
        ax.annotate(f'Peak: {max_val:,} datasets\nat {max_idx} months',
                    xy=(max_idx, max_val),
                    xytext=(max_idx + 10, max_val + max_val * 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save_name:
            self._ensure_plots()
            plt.savefig(self.plots / save_name, dpi=300, bbox_inches='tight')
            print(f"Line chart for publication_age saved to {self.plots / save_name}")
        else:
            plt.show()

    def _ensure_plots(self):
        self.plots.mkdir(exist_ok=True)
        
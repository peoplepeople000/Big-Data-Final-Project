from domain import Domain

def main():
    nola = Domain("data.nola.gov")
    # nola.download_all_raw_dataset()
    print(nola.fetch_schema("x5fx-4tmu"))
    print(nola.fetch_schema("devm-es8b"))
    print(nola.fetch_schema("em4n-zidu"))

if '__main__':
    main()
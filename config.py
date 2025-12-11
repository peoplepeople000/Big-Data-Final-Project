DATASETS = [
    {
        "name": "311",
        "id": "erm2-nwe9",   # 311 Service Requests dataset
        "domain_folder": "data.cityofnewyork.us",
        "columns": ["borough", "incident_zip"]
    },
    {
        "name": "collisions",
        "id": "h9gi-nx95",   # Motor Vehicle Collisions
        "domain_folder": "data.cityofnewyork.us",
        "columns": ["boro", "zip_code"]
    },
    {
        "name": "parks",
        "id": "enfh-gkve",   # NYC Parks dataset
        "domain_folder": "data.cityofnewyork.us",
        "columns": ["borough_name", "park_id"]
    }
]

USE_FAKE_DATA = False

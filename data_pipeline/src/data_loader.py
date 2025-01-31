# import yaml
# import requests
# import os

# SAVE_DIR = "data/raw/"
# os.makedirs(SAVE_DIR, exist_ok=True)

# def fetch_data_from_api(api_config):
#     """Fetch data from an API and save it to a JSON file."""
#     api_name = api_config["name"]
#     api_url = api_config["base_url"] + api_config["endpoint"]
#     headers = api_config["headers"]
#     params = api_config["params"]
#     auth = api_config.get("auth", {})

#     response = requests.get(api_url, headers=headers, params=params)
    
#     if response.status_code == 200:
#         save_path = os.path.join(SAVE_DIR, f"{api_name}.json")
#         with open(save_path, "w", encoding="utf-8") as file:
#             file.write(response.text)
#         print(f"✅ Successfully saved data from {api_name}")
#     else:
#         print(f"❌ Failed to fetch {api_name}: {response.status_code} - {response.text}")

# if __name__ == "__main__":
#     with open("config/api_config.yaml", "r") as file:
#         config = yaml.safe_load(file)

#     for api in config["api_sources"]:
#         fetch_data_from_api(api)

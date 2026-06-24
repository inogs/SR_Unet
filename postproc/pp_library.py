import os
import re

MONTHS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]

SEASONS = ["winter", "spring", "summer", "autumn"]

SEASON_IDS = {
    "winter": "01",
    "spring": "02",
    "summer": "03",
    "autumn": "04",
}

MONTH_IDS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

def make_month_file_dict():
    return {month: [] for month in MONTHS}

def make_season_file_dict():
    return {season: [] for season in SEASONS}

def extract_year_and_period_id(s):
    file_name = os.path.basename(s)
    groups = re.findall(r"\d+", file_name)
    if len(groups) < 2:
        return (None, None)
    return groups[-2], groups[-1]

def determine_season(period_id):
    t = int(period_id)
    if 0 <= t < 18:
        return "winter"
    elif 18 <= t < 36:
        return "spring"
    elif 36 <= t < 54:
        return "summer"
    elif 54 <= t <= 72:
        return "autumn"
    else:
        return None

def determine_month(period_id):
    t = int(period_id)
    if 0 <= t < 6:
        return "jan"
    elif 6 <= t < 12:
        return "feb"
    elif 12 <= t < 18:
        return "mar"
    elif 18 <= t < 24:
        return "apr"
    elif 24 <= t < 30:
        return "may"
    elif 30 <= t < 36:
        return "jun"
    elif 36 <= t < 42:
        return "jul"
    elif 42 <= t < 48:
        return "aug"
    elif 48 <= t < 54:
        return "sep"
    elif 54 <= t < 60:
        return "oct"
    elif 60 <= t < 66:
        return "nov"
    elif 66 <= t <= 72:
        return "dec"
    else:
        return None

def read_file_list(path):
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                files.append(line)
    return files

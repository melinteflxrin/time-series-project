EXCHANGE_RATE_DATASET = "ert_bil_eur_m"
EXCHANGE_RATE_FILTERS = {
    "geo": "RO",
    "currency": "RON",
}
EXCHANGE_RATE_OUTPUT = "data/eur_ron_monthly.csv"

HICP_DATASET = "prc_hicp_midx"
HICP_FILTERS = {
    "geo": "RO",
    "coicop": "CP00",
    "unit": "I15",
}
HICP_OUTPUT = "data/hicp_ro.csv"

SHORT_RATE_DATASET = "irt_st_m"
SHORT_RATE_FILTERS = {
    "geo": "RO",
}
SHORT_RATE_OUTPUT = "data/short_rate_ro.csv"

APP2_OUTPUT = "data/app2_monthly.csv"

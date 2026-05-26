# Application 1 – Romania HICP (Consumer Price Index) monthly, all items, index 2015=100
HICP_MONTHLY_DATASET = "prc_hicp_midx"
HICP_MONTHLY_FILTERS = {"geo": "RO", "coicop": "CP00", "unit": "I15"}
HICP_MONTHLY_OUTPUT = "../data/hicp_ro_monthly.csv"

# Application 2 – Romania short-term money-market interest rate, monthly
SHORT_RATE_MONTHLY_DATASET = "irt_st_m"
SHORT_RATE_MONTHLY_FILTERS = {"geo": "RO"}
SHORT_RATE_MONTHLY_OUTPUT = "../data/short_rate_ro_monthly.csv"

# Application 2 – Romania industrial production index, manufacturing, seasonally & calendar adjusted
INDPROD_MONTHLY_DATASET = "sts_inpr_m"
INDPROD_MONTHLY_FILTERS = {"geo": "RO", "s_adj": "SCA", "nace_r2": "C", "unit": "I15"}
INDPROD_MONTHLY_OUTPUT = "../data/indprod_ro_monthly.csv"

APP2_OUTPUT = "../data/app2_monthly.csv"

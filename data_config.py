"""
author: Florian Krach
"""
import numpy as np
import pandas as pd
from scipy import interpolate
import copy



# ==============================================================================
# FUNCTIONS
# ==============================================================================
def get_capacity_factor_adjusted_capacities(
        max_capacities_energy_mix, energy_types_desc,
        capacity_factor_adjusted=True):
    """
    this functions computes from the max capacities of the energy mix (can
    be given in total or percentage of total) the capacities adjusted by the
    capacity_factor for each of the power plant types (as percentage of total).
    The capacity_factor is given in the energy_types_desc dictionary and tells
    how much of the max capacity is actually available on average.


    Args:
        max_capacities_energy_mix: dict, keys are energy types, values are the
            max capacities of each energy type in the energy mix
        energy_types_desc: dict, describing the energy types, with their
            capacity_factor, etc.
        capacity_factor_adjusted: bool, whether to adjust the max capacities by
            capacity factor to get realizable capacities

    Returns:
        adjusted_max_capacities: dict, keys are energy types, values are the
            capacity-factor adjusted (if capacity_factor_adjusted=True) maximum
            capacities of each energy type in the energy mix (in percentage
            of total capacity)
        tot_capacity: float, the summed (capacity-factor adjusted) capacities
            of all power plants;
            (!) only works if the max_capacities_energy_mix are given in total,
            not in percentage of total
    """
    adj_capacities = []
    keys = []
    max_capacities_energy_mix = copy.deepcopy(max_capacities_energy_mix)
    for key, value in max_capacities_energy_mix.items():
        keys.append(key)
        if capacity_factor_adjusted:
            adj_capacities.append(
                value * energy_types_desc[key]["capacity_factor"])
        else:
            adj_capacities.append(value)
    adj_capacities = np.array(adj_capacities)
    tot_capacity = np.sum(adj_capacities)
    capacities_ratio = adj_capacities/tot_capacity
    adj_capacities_ratio = dict(
        zip(keys, capacities_ratio))
    return adj_capacities_ratio, tot_capacity


def get_capacity_and_energy_mix_for_plant_changes(
        starting_total_capacity, starting_max_capacities_energy_mix,
        years, energy_types_desc, capacity_changes=None,
        capacity_factor_adjusted=False):
    """
    This function computes the capacity and energy mix for the power plants
    over the time period incorporating the capacity changes given in the
    capacity_changes list. can be capacity-factor adjusted.

    Args:
        starting_total_capacity: float, the total capacity of all power plants
            at the beginning of the time period
        starting_max_capacities_energy_mix: dict, keys are energy types, values
            are the max capacities of each energy type in the energy mix at the
            beginning of the time period (in percentage of total capacity)
        years: list, the years of the time period
        energy_types_desc: dict, see get_capacity_factor_adjusted_capacities
        capacity_changes: dict, keys are strings of the year, values are lists
            where each entry is a dict with the keys "energy_type",
            "capacity_change", which tell which power plant type is changed by
            how much in the respective year. If None, no changes are made.
        capacity_factor_adjusted: bool, if True, the capacities are adjusted
            by the capacity factor of the respective power plant type

    Returns:
        capacities: list of floats, the total capacity of all power plants for
            each year
        energy_mix: list of dicts, each dict is the energy mix for the
            corresponding year
    """
    T = len(years)
    starting_max_capacities_energy_mix = copy.deepcopy(
        starting_max_capacities_energy_mix)

    # get capacity for each year
    capacities = []
    energy_mix = []
    for key, value in starting_max_capacities_energy_mix.items():
        starting_max_capacities_energy_mix[key] = value*starting_total_capacity
    if capacity_changes is None:
        capacity_changes = {}
    tot_cap_per_energy_type = {}
    last = copy.deepcopy(starting_max_capacities_energy_mix)
    for year in years:
        new = copy.deepcopy(last)
        if str(year) in capacity_changes:
            change_list = capacity_changes[str(year)]
            for change in change_list:
                new[change["energy_type"]] = max(
                    0, new[change["energy_type"]] + change["capacity_change"])
        tot_cap_per_energy_type[str(year)] = new
        last = new
    for year in years:
        e_mix, tot_cap = get_capacity_factor_adjusted_capacities(
            max_capacities_energy_mix=tot_cap_per_energy_type[str(year)],
            energy_types_desc=energy_types_desc,
            capacity_factor_adjusted=capacity_factor_adjusted)
        capacities.append(tot_cap)
        energy_mix.append(e_mix)
    return capacities, energy_mix


def get_default_prob_from_default_intensity(
        default_intensities, R=0.4, **kwargs):
    """
    this function computes the default probability from the default intensity
    and the maturity (assuming that the hazard rate is constant).

    Args:
        default_intensities: np.array, dim [maturity+1,], the default intensities
            for each maturity (in years). i.e. default_intensities[t+1] is the
            default intensity for maturity t
        R: float, recovery rate
    """
    maturities = np.arange(len(default_intensities))
    return np.minimum(1/(1-R) * (1 - np.exp(-default_intensities*maturities)), 1.)



# ==============================================================================
# ALL NEEDED DATA FROM MARKET ETC.
# ==============================================================================
TMAX = 100


# ------ CONVERSION RATES -------
usd_to_zar_Apr2022 = 14.5954  # FX rate on 2023-01-01 (=> all based on ZAR2023)
gallons_to_liters = 3.78541
pounds_to_kg = 0.453592
BTU_to_kWH = 0.000293071
kg_per_mBTU_to_kg_per_kWh = 1/(BTU_to_kWH*10**6)
usdJan2010_to_usdApr2022 = 1.33  # https://www.bls.gov/data/inflation_calculator.htm
usdSep2021_to_usdApr2022 = 1.05  # https://www.bls.gov/data/inflation_calculator.htm
gas_kg_to_kwh = 13.6  # https://www.elgas.com.au/blog/389-lpg-conversions-kg-litres-mj-kwh-and-m3/#:~:text=Convert%20LPG%20kg%20to%20kWh,kWh%20of%20energy%20from%20LPG
MW_to_KW = 10**3
hours_per_year = 365*24


# ------ ENERGY TYPES -------
# CO2e emissions taken from
#   https://www.eia.gov/environment/emissions/co2_vol_mass.php
# fuel amount taken from
#   https://www.eia.gov/tools/faqs/faq.php?id=667&t=2
# for nuclear taken from:
#   https://world-nuclear.org/information-library/economic-aspects/economics-of-nuclear-power.aspx
# the yearly_depriciation_per_capacity_unit is discounted to 2023 and taken from
#   "The carbon equivalence principle: methods and applications" Table 1
ENERGY_TYPES_DESC = dict(
    # amount_unit=[kg], emission_unit=[kg], energy_unit=[kWh],
    # capacity_unit=[kW]
    coal=dict(
        amount_per_energy_unit=0.61,
        emission_per_energy_unit=96.10*kg_per_mBTU_to_kg_per_kWh,
        yearly_depriciation_per_capacity_unit=2.552*10**9/(
                650*MW_to_KW)/40*usd_to_zar_Apr2022,
        capital_cost=2.552*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.85,
      maintenance_cost_per_capacity_unit=0.0),
    # amount_unit=[kg], emission_unit=[kg], energy_unit=[kWh],
    # capacity_unit=[kW]
    clean_coal=dict(
        amount_per_energy_unit=0.61,
        emission_per_energy_unit=0.1*96.10*kg_per_mBTU_to_kg_per_kWh,
        yearly_depriciation_per_capacity_unit=4.079*10**9/(
                650*MW_to_KW)/40*usd_to_zar_Apr2022,
        capital_cost=4.079*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.85,
        maintenance_cost_per_capacity_unit=0.0),
    # amount_unit=[kg], emission_unit=[kg], energy_unit=[kWh],
    # capacity_unit=[kW]
    nuclear=dict(
        amount_per_energy_unit=1/360000,
        emission_per_energy_unit=0,
        yearly_depriciation_per_capacity_unit=3.967*10**9/(
                600*MW_to_KW)/40*usd_to_zar_Apr2022,
        capital_cost=3.967*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.9,
        maintenance_cost_per_capacity_unit=0.0),
    # amount_unit=[KWh], emission_unit=[kg], energy_unit=[kWh],
    # capacity_unit=[kW]
    gas=dict(
        amount_per_energy_unit=1.,
        emission_per_energy_unit=52.91*kg_per_mBTU_to_kg_per_kWh,
        yearly_depriciation_per_capacity_unit=0.484*10**9/(
                418*MW_to_KW)/40*usd_to_zar_Apr2022,
        capital_cost=0.484*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.87,
        maintenance_cost_per_capacity_unit=0.0),
    clean_gas=dict(
        amount_per_energy_unit=1.,
        emission_per_energy_unit=0.1*52.91*kg_per_mBTU_to_kg_per_kWh,
        yearly_depriciation_per_capacity_unit=0.999*10**9/(
                377*MW_to_KW)/40*usd_to_zar_Apr2022,
        capital_cost=0.999*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.87,
        maintenance_cost_per_capacity_unit=0.0),
    hydro=dict(
        amount_per_energy_unit=0.0, emission_per_energy_unit=0,
        yearly_depriciation_per_capacity_unit=0.568*10**9/(
                100*MW_to_KW)/50*usd_to_zar_Apr2022,
        capital_cost=0.568*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.5,
        maintenance_cost_per_capacity_unit=0.0),
    wind_onshore=dict(
        amount_per_energy_unit=0.0, emission_per_energy_unit=0,
        yearly_depriciation_per_capacity_unit=0.270*10**9/(
                200*MW_to_KW)/25*usd_to_zar_Apr2022,
        capital_cost=0.270*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.38,
        maintenance_cost_per_capacity_unit=0.0),
    wind_offshore=dict(
        amount_per_energy_unit=0.0, emission_per_energy_unit=0,
        yearly_depriciation_per_capacity_unit=1.869*10**9/(
                400*MW_to_KW)/25*usd_to_zar_Apr2022,
        capital_cost=1.869*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.39,
        maintenance_cost_per_capacity_unit=0.0),
    solar=dict(
        amount_per_energy_unit=0.0, emission_per_energy_unit=0,
        yearly_depriciation_per_capacity_unit=0.210*10**9/(
                150*MW_to_KW)/30*usd_to_zar_Apr2022,
        capital_cost=0.210*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.158,
        maintenance_cost_per_capacity_unit=0.0),
    # amount_unit=[liter], emission_unit=[kg], energy_unit=[kWh],
    # capacity_unit=[kW]
    diesel=dict(
        amount_per_energy_unit=0.08*gallons_to_liters,
        emission_per_energy_unit=74.14*kg_per_mBTU_to_kg_per_kWh,
        # same as for gas, since ESKOM uses the same turbines
        yearly_depriciation_per_capacity_unit=0.484*10**9/(
                418*MW_to_KW*0.870)/40*usd_to_zar_Apr2022,
        capital_cost=0.484*10**9*usd_to_zar_Apr2022,
        capacity_factor=0.87,
        maintenance_cost_per_capacity_unit=0.0),
)
ENERGY_TYPES = sorted(list(ENERGY_TYPES_DESC.keys()))

ENERGY_TYPES_NAMES_DICT = {
    "coal": "Coal",
    "clean_coal": "Coal (CCS90)",
    "nuclear": "Nuclear",
    "gas": "Gas",
    "clean_gas": "Gas (CCS90)",
    "hydro": "Hydro",
    "wind_onshore": "Wind (onshore)",
    "wind_offshore": "Wind (offshore)",
    "solar": "Solar",
    "diesel": "Diesel",
}


# ------ INTEREST AND INFLATION FACTORS -------
# south african inflation and interest rates
df_interest = pd.read_csv('data/interest_rate_term_structure.csv',
                          index_col=None)
df_interest["date"] = pd.to_datetime(df_interest["date"]).astype(int)/10**9
years_str = pd.DataFrame(
    {"date": ["01.04.{}".format(x) for x in np.arange(2022,2051)]})
years = pd.to_datetime(years_str["date"]).astype(int).values/10**9
f = interpolate.interp1d(df_interest["date"].values, df_interest["rate"].values)
interest_rate = f(years[:-1])
INV_DISCOUNTING_FACTOR = np.cumprod(1+interest_rate/100)
df_real = pd.read_csv('data/real_rate_term_structure.csv', index_col=None)
df_real["date"] = pd.to_datetime(df_real["date"]).astype(int)/10**9
f = interpolate.interp1d(df_real["date"].values, df_real["rate"].values)
real_rate = f(years[:-1])
INFLATION_FACTOR = np.cumprod(1+(interest_rate-real_rate)/100)

# credit ratings expressed in default probabilities
df_credit_ratings = pd.read_csv('data/credit_ratings.csv', index_col=0)
default_prob_credit_rating_curves = dict()
for name in df_credit_ratings.index[1:]:
    spreads = \
        [df_credit_ratings.loc[name,"5Y"]-
         df_credit_ratings.loc["ref","5Y"]]*10+ \
        [df_credit_ratings.loc[name,"10Y"]-
         df_credit_ratings.loc["ref","10Y"]]*10+\
        [df_credit_ratings.loc[name,"20Y"]-
         df_credit_ratings.loc["ref","20Y"]]*10
    spreads = np.array(spreads)
    default_prob_credit_rating_curves[name] = \
        get_default_prob_from_default_intensity(spreads, R=0.4)
credit_ratings_to_use = ["AAA", "AA", "A", "BBB", "BB", "B", "B-"]
DEFAULT_PROB_CREDIT_RATING_CURVES = dict()
for name in credit_ratings_to_use:
    DEFAULT_PROB_CREDIT_RATING_CURVES[name] = \
        default_prob_credit_rating_curves[name]

# us interest and inflation factors
df_interest_us = pd.read_csv(
    'data/us_interest_rate.csv', index_col=None)
df_interest_us["date"] = pd.to_datetime(
    df_interest_us["date"]).astype(int)/10**9
f = interpolate.interp1d(
    df_interest_us["date"].values, df_interest_us["rate"].values)
us_interest_rate = f(years[:-1])
US_INV_DISCOUNTING_FACTOR = np.cumprod(1+us_interest_rate/100)
df_real_us = pd.read_csv('data/us_real_rate.csv', index_col=None)
df_real_us["date"] = pd.to_datetime(df_real_us["date"]).astype(int)/10**9
f = interpolate.interp1d(df_real_us["date"].values, df_real_us["rate"].values)
us_real_rate = f(years[:-1])
US_INFLATION_FACTOR = np.cumprod(1+(us_interest_rate-us_real_rate)/100)



# ------ NGFS (& OTHER) CO2 PRICE SCENARIOS -------
NO_COST_SCENARIO = [0.]*TMAX
ZA_CARBON_TAX_SCENARIO = [0.159] + list(
    0.159 * INFLATION_FACTOR/INV_DISCOUNTING_FACTOR)  # ZAR/kg
df = pd.read_csv('data/ngfs_scenarios.csv', index_col=0)
scenario_names = df.index.tolist()
YEARS = np.arange(2022, 2051, 1)
T50 = len(YEARS)-1
EMISSION_PRICE_SCENRIOS = {
    "no_cost": NO_COST_SCENARIO[:T50],
    "South African Carbon Tax": ZA_CARBON_TAX_SCENARIO[:T50]}
for scenario_name in scenario_names:
    prices = df.loc[scenario_name]*usd_to_zar_Apr2022*\
             usdJan2010_to_usdApr2022*10**-3
    times = [int(x) for x in df.columns.tolist()]
    f = interpolate.interp1d(times, prices)
    EMISSION_PRICE_SCENRIOS[scenario_name] = \
        f(YEARS[:-1])/US_INV_DISCOUNTING_FACTOR

NGFS_SCENARIO_NAMES_DICT = {
    "no_cost": "No Cost",
    "South African Carbon Tax": "SA Carbon Tax",
    "Delayed transition": "Delayed Transition",
    "Divergent Net Zero": "Divergent Net Zero",
    "Current Policies": "Current Policies",
    "Nationally Determined Contributions (NDCs)": "NDCs",
    "Net Zero 2050": "Net Zero 2050",
    "Below 2C": "Below 2C",}






# ------ FUEL PRICES -------
# nuclear fuel price taken from
#  https://world-nuclear.org/information-library/economic-aspects/economics-of-nuclear-power.aspx
# ZA gas price taken as average from
#  https://www.energy.gov.za/files/esources/petroleum/July2023/LPG-Regulations.pdf
# in the following we divide by first inflation factor to have exactly this price in first year
current_nuclear_price = 1663*usd_to_zar_Apr2022*usdSep2021_to_usdApr2022/\
                        INFLATION_FACTOR[0]  # in ZAR/kg
current_gas_price = 3111.796/100/gas_kg_to_kwh/INFLATION_FACTOR[0]  # in ZAR/kWh
eskom_coal_price = 69.99*10**9/(170514*10**6*0.61)/INFLATION_FACTOR[0]  # in ZAR/kg, from table 72 and burn rate from 8.7.16 in https://www.nersa.org.za/wp-content/uploads/bsk-pdf-manager/2023/02/Eskoms-MYPD5-RfD-for-202324FY-and-202425FY_Public-Version.pdf
eskom_diesel_price = np.mean((23.51,20.36,20.28,20.22))/INFLATION_FACTOR[0] # mean of eskom's curent normal contact prices stated in https://businesstech.co.za/news/energy/661635/diesel-sharks-smell-blood-at-eskom/

FUEL_PRICES_PER_AMOUNT_UNIT = dict(
    # ESKOM pays 1/2.257 of actual coal price
    coal=eskom_coal_price*INFLATION_FACTOR/INV_DISCOUNTING_FACTOR,
    clean_coal=eskom_coal_price*INFLATION_FACTOR/INV_DISCOUNTING_FACTOR,
    nuclear=current_nuclear_price*US_INFLATION_FACTOR/US_INV_DISCOUNTING_FACTOR,
    gas=current_gas_price*US_INFLATION_FACTOR/US_INV_DISCOUNTING_FACTOR,
    clean_gas=current_gas_price*US_INFLATION_FACTOR/US_INV_DISCOUNTING_FACTOR,
    hydro=[0.]*TMAX,
    wind_onshore=[0.]*TMAX,
    wind_offshore=[0.]*TMAX,
    solar=[0.]*TMAX,
    diesel=eskom_diesel_price*US_INFLATION_FACTOR/US_INV_DISCOUNTING_FACTOR,
)


# ------ NET PRICES -------
# the following is meant as the emission reduction per time step per price unit,
#   it should be adjusted for inflation, but not discounted (discounting is
#   done through the discounted emission price)
NET_EMISSION_REDUCTION_PER_PRICE_UNIT = [0.]*TMAX



# ------ ESKOM VALUES -------
CURRENT_EQUITY = float(235314*10**6)  # in ZAR
CURRENT_YEARLY_SOLD_ENERGY = float(192.005*10**9)  # in kWh
CURRENT_ELECTRICITY_PRICE = 1.2275 * 1.0961  # in ZAR/kWh
CURRENT_DEPRECIATION_COSTS = 32009*10**6  # in ZAR, not used as included in ENERGY_DESC
maintenance_costs = 24113*10**6  # in ZAR
labour_costs = 32.985*10**9
RUNNING_COSTS = \
    float(maintenance_costs+labour_costs) * \
    INFLATION_FACTOR / INV_DISCOUNTING_FACTOR
eskom_co2e_emissions_2021_2022 = 207230321 *10**3  # in kg
environmental_levy_2021 = 7191*10**6  # in ZAR
env_levy_per_co2e = environmental_levy_2021/eskom_co2e_emissions_2021_2022  # ZAR/kg
additional_energy_consumed_by_pumped_storage = \
    2732*MW_to_KW*hours_per_year*0.17*0.3

# default probability
df_eskom_survival = pd.read_csv('data/default_probs_eskom.csv', index_col=None)
df_eskom_survival["date"] = pd.to_datetime(
    df_eskom_survival["date"]).astype(int)/10**9
f = interpolate.interp1d(
    df_eskom_survival["date"].values, df_eskom_survival["ND prob"].values)
eskom_survival_probs = f(years)
ESKOM_DEFAULT_PROBS = 1-eskom_survival_probs

# energy mix
CURRENT_ENERGY_MIX_MAX_CAPACITY = dict(
    # all in MW
    coal=44013.0,
    clean_coal=0.0,
    hydro=661.4,
    wind_onshore=100.0,
    wind_offshore=0.0,
    solar=0.0,
    gas=342.0,
    clean_gas=0.0,
    diesel=2078.3,
    nuclear=1934.0,
)
CURRENT_ENERGY_MIX_MAX_RATIOS, current_total_max_capacity = \
    get_capacity_factor_adjusted_capacities(
        CURRENT_ENERGY_MIX_MAX_CAPACITY,
        energy_types_desc=ENERGY_TYPES_DESC,
        capacity_factor_adjusted=False)
CURRENT_YEARLY_MAX_GENERATION_CAPACITY = \
    current_total_max_capacity*MW_to_KW  # in kW

CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ, CURRENT_REALIZABLE_CAPACITY = \
    get_capacity_factor_adjusted_capacities(
        max_capacities_energy_mix=CURRENT_ENERGY_MIX_MAX_CAPACITY,
        energy_types_desc=ENERGY_TYPES_DESC,
        capacity_factor_adjusted=True)
current_energy_mix_total_realizeable_capacities = dict()
for k, v in CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ.items():
    current_energy_mix_total_realizeable_capacities[k] = \
        v*CURRENT_REALIZABLE_CAPACITY

# overall and coal power production factor (assuming that coal is the only
#   energy source with a ppf<1)
power_production_factor = \
    CURRENT_YEARLY_SOLD_ENERGY/\
    (CURRENT_REALIZABLE_CAPACITY*MW_to_KW*hours_per_year)
coal_power_production_factor = \
    (power_production_factor -
     (1-CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ["coal"]))/\
    CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ["coal"]
# power production factor if pumped storage energy use is considered
power_production_factor1 = \
    (CURRENT_YEARLY_SOLD_ENERGY+additional_energy_consumed_by_pumped_storage)/\
    (CURRENT_REALIZABLE_CAPACITY*MW_to_KW*hours_per_year)
coal_power_production_factor1 = \
    (power_production_factor1 -
     (1-CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ["coal"]))/\
    CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ["coal"]

# dict with current power production factors
CURRENT_POWER_PRODUCTION_FACTORS = dict()
for k, v in ENERGY_TYPES_DESC.items():
    CURRENT_POWER_PRODUCTION_FACTORS[k] = 1.
CURRENT_POWER_PRODUCTION_FACTORS["coal"] = coal_power_production_factor
CURRENT_POWER_PRODUCTION_FACTORS["clean_coal"] = coal_power_production_factor

# emissions factor
model_yearly_emissions = np.sum([
    CURRENT_POWER_PRODUCTION_FACTORS[energy_type]*
    CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ[energy_type]*
    CURRENT_YEARLY_SOLD_ENERGY*ENERGY_TYPES_DESC[energy_type][
        "emission_per_energy_unit"]
    for energy_type in ENERGY_TYPES_DESC.keys()])
EMISSIONS_FACTOR = eskom_co2e_emissions_2021_2022/model_yearly_emissions

MAX_CAPACITY_BASE_CASE, MAX_CAPACITY_BASE_CASE_ENERGY_MIX = \
    get_capacity_and_energy_mix_for_plant_changes(
        starting_total_capacity=CURRENT_YEARLY_MAX_GENERATION_CAPACITY,
        starting_max_capacities_energy_mix=CURRENT_ENERGY_MIX_MAX_RATIOS,
        years=YEARS[:T50], energy_types_desc=ENERGY_TYPES_DESC,
        capacity_changes=None, capacity_factor_adjusted=False)

model_yearly_depreciation = np.sum([
    MAX_CAPACITY_BASE_CASE_ENERGY_MIX[0][energy_type]*
    MAX_CAPACITY_BASE_CASE[0]*ENERGY_TYPES_DESC[energy_type][
        "yearly_depriciation_per_capacity_unit"]
    for energy_type in ENERGY_TYPES_DESC])
DEPRECIATION_COST_FACTOR = CURRENT_DEPRECIATION_COSTS/model_yearly_depreciation

POWER_PRODUCTION_FACTORS = [CURRENT_POWER_PRODUCTION_FACTORS]*T50

# Capacity and energy mix scenario based on integrated resource plan for ZA
#   https://www.energy.gov.za/irp/2019/IRP-2019.pdf p.42
IRP2030_CAPACITY_CHANGES = {
    "2023": [],
    "2024": [
        {"energy_type": "coal", "capacity_change": 195*MW_to_KW},
        {"energy_type": "solar", "capacity_change": 1000*MW_to_KW},
        {"energy_type": "wind_onshore",
         "capacity_change": 1600*MW_to_KW},],
    "2025": [
        {"energy_type": "nuclear", "capacity_change": 1860*MW_to_KW},
        {"energy_type": "wind_onshore",
         "capacity_change": 1600*MW_to_KW},
        {"energy_type": "gas", "capacity_change": 1000*MW_to_KW},],
    "2026": [
        {"energy_type": "wind_onshore",
         "capacity_change": 1600 * MW_to_KW},
        {"energy_type": "solar", "capacity_change": 1000*MW_to_KW},],
    "2027": [
        {"energy_type": "coal", "capacity_change": -1219*MW_to_KW},
        {"energy_type": "wind_onshore",
         "capacity_change": 1600 * MW_to_KW},],
    "2028": [
        {"energy_type": "coal", "capacity_change": -97*MW_to_KW},
        {"energy_type": "wind_onshore",
         "capacity_change": 1600 * MW_to_KW},
        {"energy_type": "gas", "capacity_change": 2000*MW_to_KW},],
    "2029": [
        {"energy_type": "coal", "capacity_change": -475*MW_to_KW},
        {"energy_type": "solar", "capacity_change": 1000*MW_to_KW},
        {"energy_type": "wind_onshore",
         "capacity_change": 1600 * MW_to_KW},],
    "2030": [
        {"energy_type": "coal", "capacity_change": -1694*MW_to_KW},
        {"energy_type": "solar", "capacity_change": 1000*MW_to_KW},
        {"energy_type": "wind_onshore",
         "capacity_change": 1600 * MW_to_KW},],
    "2031": [
        {"energy_type": "coal", "capacity_change": -1050*MW_to_KW},
        {"energy_type": "hydro", "capacity_change": 2500*MW_to_KW},
        {"energy_type": "solar", "capacity_change": 1000*MW_to_KW},
        {"energy_type": "wind_onshore",
         "capacity_change": 1600 * MW_to_KW},],
}
IRP2030_CAPACITIES, IRP2030_ENERGY_MIX = \
    get_capacity_and_energy_mix_for_plant_changes(
        starting_total_capacity=CURRENT_YEARLY_MAX_GENERATION_CAPACITY,
        starting_max_capacities_energy_mix=CURRENT_ENERGY_MIX_MAX_RATIOS,
        years=YEARS[:T50], energy_types_desc=ENERGY_TYPES_DESC,
        capacity_changes=IRP2030_CAPACITY_CHANGES,
        capacity_factor_adjusted=False,)
IRP2030_REALIZABLE_CAPACITIES, IRP2030_REALIZABLE_ENERGY_MIX = \
    get_capacity_and_energy_mix_for_plant_changes(
        starting_total_capacity=CURRENT_YEARLY_MAX_GENERATION_CAPACITY,
        starting_max_capacities_energy_mix=CURRENT_ENERGY_MIX_MAX_RATIOS,
        years=YEARS[:T50], energy_types_desc=ENERGY_TYPES_DESC,
        capacity_changes=IRP2030_CAPACITY_CHANGES,
        capacity_factor_adjusted=True,)

# green continuation of IRP2030
IRP2030_CONTINUED_CAPACITY_CHANGES = copy.deepcopy(IRP2030_CAPACITY_CHANGES)
continued_yearly_change = [
    {"energy_type": "coal", "capacity_change": -1073*MW_to_KW},
    {"energy_type": "solar", "capacity_change": 1000*MW_to_KW},
    {"energy_type": "wind_onshore",
     "capacity_change": 1600 * MW_to_KW},]
for year in np.arange(2032, 2051, 1):
    IRP2030_CONTINUED_CAPACITY_CHANGES[str(year)] = copy.deepcopy(
        continued_yearly_change)
for year in [2034, 2044]:
    IRP2030_CONTINUED_CAPACITY_CHANGES[str(year)] += [
        {"energy_type": "nuclear", "capacity_change": 1860*MW_to_KW},]
for year in [2037, 2047]:
    IRP2030_CONTINUED_CAPACITY_CHANGES[str(year)] += [
        {"energy_type": "hydro", "capacity_change": 2500*MW_to_KW},]
IRP2030_CONTINUED_CAPACITIES, IRP2030_CONTINUED_ENERGY_MIX = \
    get_capacity_and_energy_mix_for_plant_changes(
        starting_total_capacity=CURRENT_YEARLY_MAX_GENERATION_CAPACITY,
        starting_max_capacities_energy_mix=CURRENT_ENERGY_MIX_MAX_RATIOS,
        years=YEARS[:T50], energy_types_desc=ENERGY_TYPES_DESC,
        capacity_changes=IRP2030_CONTINUED_CAPACITY_CHANGES,
        capacity_factor_adjusted=False,)
IRP2030_CONTINUED_REALIZABLE_CAPACITIES, \
IRP2030_CONTINUED_REALIZABLE_ENERGY_MIX = \
    get_capacity_and_energy_mix_for_plant_changes(
        starting_total_capacity=CURRENT_YEARLY_MAX_GENERATION_CAPACITY,
        starting_max_capacities_energy_mix=CURRENT_ENERGY_MIX_MAX_RATIOS,
        years=YEARS[:T50], energy_types_desc=ENERGY_TYPES_DESC,
        capacity_changes=IRP2030_CONTINUED_CAPACITY_CHANGES,
        capacity_factor_adjusted=True,)

# aggressive green continuation of IRP2030
IRP2030_CONTINUED_CAPACITY_CHANGES_3 = copy.deepcopy(IRP2030_CAPACITY_CHANGES)
for k, v in IRP2030_CONTINUED_CAPACITY_CHANGES_3.items():
    for d in v:
        if d["energy_type"] == "gas":
            d["energy_type"] = "clean_gas"
        elif d["energy_type"] == "diesel":
            d["energy_type"] = "clean_gas"
for year in np.arange(2023, 2030, 1):
    IRP2030_CONTINUED_CAPACITY_CHANGES_3[str(year)] += [
        {"energy_type": "clean_coal", "capacity_change": 5500 * MW_to_KW},
        {"energy_type": "coal", "capacity_change": -6000 * MW_to_KW},
    ]
    if year in [2023, 2024, 2025, 2026]:
        IRP2030_CONTINUED_CAPACITY_CHANGES_3[str(year)] += [
            {"energy_type": "diesel", "capacity_change": -2078.3/4. * MW_to_KW},
            {"energy_type": "clean_gas", "capacity_change": 2078.3/4. * MW_to_KW},
            {"energy_type": "gas", "capacity_change": -342/4. * MW_to_KW},
            {"energy_type": "clean_gas", "capacity_change": 342/4. * MW_to_KW},
        ]
continued_yearly_change_3 = [
    {"energy_type": "clean_coal", "capacity_change": -1073*MW_to_KW},
    {"energy_type": "solar", "capacity_change": 1000*MW_to_KW},
    {"energy_type": "wind_onshore",
     "capacity_change": 1600 * MW_to_KW},
]
for year in np.arange(2032, 2051, 1):
    IRP2030_CONTINUED_CAPACITY_CHANGES_3[str(year)] = copy.deepcopy(
        continued_yearly_change_3)
for year in [2034, 2044]:
    IRP2030_CONTINUED_CAPACITY_CHANGES_3[str(year)] += [
        {"energy_type": "nuclear", "capacity_change": 1860*MW_to_KW},]
for year in [2037, 2047]:
    IRP2030_CONTINUED_CAPACITY_CHANGES[str(year)] += [
        {"energy_type": "hydro", "capacity_change": 2500*MW_to_KW},]

IRP2030_CONTINUED_CAPACITIES_3, IRP2030_CONTINUED_ENERGY_MIX_3 = \
    get_capacity_and_energy_mix_for_plant_changes(
        starting_total_capacity=CURRENT_YEARLY_MAX_GENERATION_CAPACITY,
        starting_max_capacities_energy_mix=CURRENT_ENERGY_MIX_MAX_RATIOS,
        years=YEARS[:T50], energy_types_desc=ENERGY_TYPES_DESC,
        capacity_changes=IRP2030_CONTINUED_CAPACITY_CHANGES_3,
        capacity_factor_adjusted=False,)
IRP2030_CONTINUED_REALIZABLE_CAPACITIES_3, \
IRP2030_CONTINUED_REALIZABLE_ENERGY_MIX_3 = \
    get_capacity_and_energy_mix_for_plant_changes(
        starting_total_capacity=CURRENT_YEARLY_MAX_GENERATION_CAPACITY,
        starting_max_capacities_energy_mix=CURRENT_ENERGY_MIX_MAX_RATIOS,
        years=YEARS[:T50], energy_types_desc=ENERGY_TYPES_DESC,
        capacity_changes=IRP2030_CONTINUED_CAPACITY_CHANGES_3,
        capacity_factor_adjusted=True,)


# ==============================================================================
# CONFIGURATIONS TO RUN
# ==============================================================================
# ------------- Currently sold capacity base case (no changes)------------------
current_sold_capacity_config_1 = dict(
    maturity=T50,
    max_maturity=T50,
    init_assets=CURRENT_EQUITY,
    init_liabilities=0,
    inv_discounting_factor=INV_DISCOUNTING_FACTOR,
    inflation_factor=INFLATION_FACTOR,
    capacities=CURRENT_YEARLY_SOLD_ENERGY,
    energy_mix=CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ,
    emission_prices=ZA_CARBON_TAX_SCENARIO,
    scenarios_emission_prices=EMISSION_PRICE_SCENRIOS,
    NET_investments=0.,
    fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT,
    running_costs=RUNNING_COSTS,
    init_capital_spending="no",
    inflation_adjusted=True,
    investments=None,
    default_probs=ESKOM_DEFAULT_PROBS,
    plot=True,
    use_running_default_probs=True,
    nb_paths=10000, seed=0,
    init_electricity_price=CURRENT_ELECTRICITY_PRICE,
    timegrid=YEARS,
    mean_percentage_jump=0.096,
    plot_scenarios_only=False,
    mean_percentages_to_plot=[0.05, 0.1, 0.2],
    plot_mppj_std=True,
    plot_postfix="setting1",
    which_cap_type_to_plot="realizable",
    default_prob_credit_ratings=DEFAULT_PROB_CREDIT_RATING_CURVES,
    us_inv_discounting_factor=US_INV_DISCOUNTING_FACTOR,
    power_production_factor=POWER_PRODUCTION_FACTORS,
    time_amount_per_step=hours_per_year,
    emissions_factor=EMISSIONS_FACTOR,
    depreciation_factor=DEPRECIATION_COST_FACTOR,
    electricity_price_scaling_factor=np.array([1.]*28),
    first_fixed=True,
    fit_mppj=True,
    fit_init_assets=False,
    optimization_kwargs=dict(
        method="Nelder-Mead", tol=None, options={"maxiter": 2000}),
)

ngfs_plot_config = copy.deepcopy(current_sold_capacity_config_1)
ngfs_plot_config["plot_scenarios_only"] = True



# -------------- IRP2030 base case (no changes afterwards) ---------------------
IRP2030_config_0 = copy.deepcopy(current_sold_capacity_config_1)
IRP2030_config_0["capacities"] = IRP2030_CAPACITIES
IRP2030_config_0["energy_mix"] = IRP2030_ENERGY_MIX
IRP2030_config_0["running_costs"] = \
    RUNNING_COSTS * np.array(IRP2030_REALIZABLE_CAPACITIES) / \
    IRP2030_REALIZABLE_CAPACITIES[0]
IRP2030_config_0["plot_postfix"] = "IRP2030-base-case"

# ----- calibrate model: init assets fixed -----
# fit 1:
IRP2030_config_1 = copy.deepcopy(IRP2030_config_0)
IRP2030_config_1["mean_percentage_jump"] = 0.094
IRP2030_config_1["fit_init_assets"] = False
IRP2030_config_1["electricity_price_scaling_factor"] = np.array([1.]*28)
IRP2030_config_1["optimization_kwargs"] = dict(
    method="Nelder-Mead", tol=None, options={"maxiter": 2000})
IRP2030_config_1["plot_postfix"] = "IRP2030-base-case-fit1"

# fit 2:
IRP2030_config_2 = copy.deepcopy(IRP2030_config_1)
IRP2030_config_2["mean_percentage_jump"] = 0.061
IRP2030_config_2["electricity_price_scaling_factor"] = np.array([1.]*5+[1.1]*23)
IRP2030_config_2["plot_postfix"] = "IRP2030-base-case-fit2"

# ----- calibrate model: init assets fixed to half of the reported equity -----
# fit 3:
IRP2030_config_3 = copy.deepcopy(IRP2030_config_0)
IRP2030_config_3["mean_percentage_jump"] = 0.094
IRP2030_config_3["fit_init_assets"] = False
IRP2030_config_3["init_assets"] = CURRENT_EQUITY*0.5
IRP2030_config_3["fit_mppj"] = True
IRP2030_config_3["electricity_price_scaling_factor"] = np.array([1.]*28)
IRP2030_config_3["optimization_kwargs"] = dict(
    method="Nelder-Mead", tol=None, options={"maxiter": 2000})
IRP2030_config_3["plot_postfix"] = "IRP2030-base-case-fit3"

# fit 4:
IRP2030_config_4 = copy.deepcopy(IRP2030_config_0)
IRP2030_config_4["mean_percentage_jump"] = 0.06
IRP2030_config_4["init_assets"] = CURRENT_EQUITY*0.5
IRP2030_config_4["fit_init_assets"] = True
IRP2030_config_4["fit_mppj"] = True
IRP2030_config_4["electricity_price_scaling_factor"] = \
    np.array([1., 1.2, 1.4, 1.4] + [1.2]*24)
IRP2030_config_4["optimization_kwargs"] = dict(
    method="Nelder-Mead", tol=None, options={"maxiter": 2000})
IRP2030_config_4["plot_postfix"] = "IRP2030-base-case-fit4"

# fit 5:
IRP2030_config_5 = copy.deepcopy(IRP2030_config_0)
IRP2030_config_5["mean_percentage_jump"] = 0.049
IRP2030_config_5["init_assets"] = CURRENT_EQUITY
IRP2030_config_5["fit_init_assets"] = 0.5
IRP2030_config_5["fit_mppj"] = True
IRP2030_config_5["electricity_price_scaling_factor"] = \
    np.array([
        1.,1.12673805,1.34653936,1.2925787,1.25516392,1.23180465,1.2564611,
        1.22921723,1.20939743,1.13953121,1.15812814,1.17836633,1.14980597,
        1.22058136,1.19920144,1.23956927,1.19517954,1.2390811,1.23427301,
        1.22023396,1.20133923,1.27200831,1.24524854,1.22877642,1.21335901,
        1.29130182,1.20906844,1.28186217])
IRP2030_config_5["optimization_kwargs"] = dict(
    method="Nelder-Mead", tol=None, options={"maxiter": 2000})
IRP2030_config_5["plot_postfix"] = "IRP2030-base-case-fit5"



# ---- fitted model
IRP2030_config_market_cal = copy.deepcopy(IRP2030_config_0)
IRP2030_config_market_cal["mean_percentage_jump"] = 0.04911192763532417
IRP2030_config_market_cal["init_assets"] = CURRENT_EQUITY*0.5000223205212551
IRP2030_config_market_cal["electricity_price_scaling_factor"] = \
    np.array([
        1.,1.12671805,1.34679554,1.29171012,1.25755471,1.22488008,1.25443813,
        1.23972701,1.21026818,1.12856752,1.17802094,1.19647877,1.16723181,
        1.16911273,1.17013379,1.21590257,1.22494943,1.20458228,1.1878981,
        1.19654374,1.31611408,1.32397877,1.29489235,1.28759401,1.20725433,
        1.30942762,1.21980351,1.25793086])
IRP2030_config_market_cal["mean_percentages_to_plot"] = [0.0556,]
IRP2030_config_market_cal["inflation_adjusted"] = True

IRP2030_config_market_cal1 = copy.deepcopy(IRP2030_config_market_cal)
IRP2030_config_market_cal1["use_running_default_probs"] = False


# -------------- IRP2030 Green continue (coal-, renewables+) -------------------
IRP2030_green_continue_config = copy.deepcopy(IRP2030_config_market_cal)
IRP2030_green_continue_config["capacities"] = IRP2030_CONTINUED_CAPACITIES
IRP2030_green_continue_config["energy_mix"] = IRP2030_CONTINUED_ENERGY_MIX
IRP2030_green_continue_config["running_costs"] = \
    RUNNING_COSTS * np.array(IRP2030_CONTINUED_REALIZABLE_CAPACITIES) / \
    IRP2030_CONTINUED_REALIZABLE_CAPACITIES[0]
IRP2030_green_continue_config["plot_postfix"] = "IRP2030-green-continuation"

IRP2030_green_continue_config1 = copy.deepcopy(IRP2030_green_continue_config)
IRP2030_green_continue_config1["use_running_default_probs"] = False


# -------- IRP2030 aggressive green continue (coal--, renewables++) -----
IRP2030_aggr_green_continue_config = copy.deepcopy(IRP2030_config_market_cal)
IRP2030_aggr_green_continue_config["capacities"] = \
    IRP2030_CONTINUED_CAPACITIES_3
IRP2030_aggr_green_continue_config["energy_mix"] = \
    IRP2030_CONTINUED_ENERGY_MIX_3
IRP2030_aggr_green_continue_config["running_costs"] = \
    RUNNING_COSTS * np.array(IRP2030_CONTINUED_REALIZABLE_CAPACITIES_3) / \
    IRP2030_CONTINUED_REALIZABLE_CAPACITIES_3[0]
IRP2030_aggr_green_continue_config["plot_postfix"] = \
    "IRP2030-aggr-green-continuation"

IRP2030_aggr_green_continue_config1 = copy.deepcopy(
    IRP2030_aggr_green_continue_config)
IRP2030_aggr_green_continue_config1["use_running_default_probs"] = False


# ---- joint plotting configs -----
joint_min_elect_price_plot_config = dict(
    plotfunc="plot_min_electricity_prices",
    configs=[
        IRP2030_config_market_cal, IRP2030_green_continue_config,
        IRP2030_aggr_green_continue_config],
    plots_per_line=3, figsize=(6.4*3/1.5, 4.8/1.5),
    suptitle_kwargs=dict(y=1.15, fontsize=16),
    legend_kwargs=dict(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05)),
    titles=["Base Case", "Green Continuation", "Green Continuation & CCS"],
    plot_suptitle=False,
)

joint_rlz_cap_plot_config = dict(
    plotfunc="plot_capacities_and_energy_mix",
    configs=[
        IRP2030_config_market_cal, IRP2030_green_continue_config,
        IRP2030_aggr_green_continue_config],
    plots_per_line=3, figsize=(6.4*3/1.5, 4.8/1.5),
    suptitle_kwargs=dict(y=1.15, fontsize=16),
    legend_kwargs=dict(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05)),
    twinx_legend_kwargs=dict(
        loc='upper center', ncol=5, bbox_to_anchor=(0.5, -0.15)),
    titles=["Base Case", "Green Continuation", "Green Continuation & CCS"],
    twinx=True,
    plot_suptitle=False,
)

joint_fitdefaultprobs_meanelecprice_plot_config = dict(
    plotfunc=["fit_electricity_price_scaling_factors",
              "plot_mean_electricity_prices_for_MPPJ"],
    configs=[IRP2030_config_5, IRP2030_config_market_cal],
    plots_per_line=2, figsize=(10*2/1.5, 4.8/1.5),
    twinx=False,
    plot_suptitle=False,
    remove_unnecessary_xlabel=False,
    remove_unnecessary_ylabel=False,
    remove_unnecessary_twinxlabel=False,
    remove_unnecessary_twinxticks=False,
    legend_outside=False,
    fname_set="results/calibrated_model_def_prob_and_mean_elec_price.pdf",
)

joint_defprob_bondprice_plot_config = dict(
    plotfunc=["plot_running_default_prob_and_scenarios", "plot_bond_price_term_structure"],
    configs=[IRP2030_config_market_cal, IRP2030_config_market_cal,],
    plots_per_line=2, figsize=(6.4*2/1.1, 4.8/1.1),
    suptitle_kwargs=dict(y=1.15, fontsize=16),
    legend_kwargs=dict(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05)),
    twinx_legend_kwargs=dict(
        labels=['$CO_2e$ price', "probability of default | bond price"],
        loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.05)),
    twinx=True,
    plot_suptitle=False,
    fname_set="results/default_prob_and_bond_price_IRP2030-base-case.pdf",
    remove_unnecessary_xlabel=False,
    remove_unnecessary_ylabel=False,
    remove_unnecessary_twinxlabel=False,
    remove_unnecessary_twinxticks=False,
    legend_outside=True,
)

joint_defprob_instdefprob_bondprice_plot_config = dict(
    plotfunc=[
        "plot_running_default_prob_and_scenarios",
        "plot_running_default_prob_and_scenarios",
        "plot_running_default_prob_and_scenarios",
        "plot_running_default_prob_and_scenarios",
        "plot_bond_price_term_structure",
        "plot_bond_price_term_structure",
    ],
    configs=[
        IRP2030_green_continue_config, IRP2030_aggr_green_continue_config,
        IRP2030_green_continue_config1, IRP2030_aggr_green_continue_config1,
        IRP2030_green_continue_config, IRP2030_aggr_green_continue_config,
    ],
    plots_per_line=2, figsize=(6.4*2/1.1, 4.8*3/1.1),
    suptitle_kwargs=dict(y=1.15, fontsize=16),
    legend_kwargs=dict(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.03)),
    twinx_legend_kwargs=dict(
        labels=[
            '$CO_2e$ price',
            "probability of default | instantaneous probability of default | "
            "bond price"
        ],
        loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.0)),
    twinx=True,
    plot_suptitle=False,
    fname_set="results/"
              "default_prob_and_bond_price_IRP2030-green-continuations.pdf",
    remove_unnecessary_xlabel=True,
    remove_unnecessary_ylabel=True,
    remove_unnecessary_twinxlabel=True,
    remove_unnecessary_twinxticks=True,
    remove_unnecessary_yticks=True,
    legend_outside=True,
)




# ------------------------------------------------------------------------------
def print_values_from_paper():
    print("values from paper:")
    print("-"*80)
    print("time step:", hours_per_year, "dates:\n", years_str)
    print("-" * 80)
    print("ZAR/USD Apr 2022:", usd_to_zar_Apr2022)
    print("USD Jan 2010 to USD Apr 2022 (inflation):", usdJan2010_to_usdApr2022)
    print("USD Sep 2021 to USD Apr 2022 (inflation):", usdSep2021_to_usdApr2022)
    print("-" * 80)
    print("inflation:\n", INFLATION_FACTOR)
    print("discounting (inverse):\n", INV_DISCOUNTING_FACTOR)
    print("US inflation:\n", US_INFLATION_FACTOR)
    print("US discounting (inverse):\n", US_INV_DISCOUNTING_FACTOR)
    print("-" * 80)
    print("Eskom market default probs:\n", ESKOM_DEFAULT_PROBS)
    print("Credit rating default probs:\n")
    for k, v in DEFAULT_PROB_CREDIT_RATING_CURVES.items():
        print("\t", k, ":", v)
    print("-" * 80)
    print("Energy types:\n")
    for k, v in ENERGY_TYPES_DESC.items():
        print("\t", k, ":",)
        for k2, v2 in v.items():
            print("\t\t", k2, ":", v2)
    print("-" * 80)
    print("capital costs for energy types in (BZAR):\n")
    for k, v in ENERGY_TYPES_DESC.items():
        print("\t", k, ":", v["capital_cost"]/10**9)
    print("-" * 80)
    print("Fuel prices:\n")
    print("Current coal price (ZAR/kg):", eskom_coal_price*INFLATION_FACTOR[0])
    print("Current gas price (ZAR/kg):",
          current_gas_price*INFLATION_FACTOR[0]*gas_kg_to_kwh)
    print("Current diesel price (ZAR/l):",
          eskom_diesel_price*INFLATION_FACTOR[0])
    print("Current nuclear price (ZAR/kg):",
          current_nuclear_price * INFLATION_FACTOR[0])
    print("-" * 80)
    for k, v in FUEL_PRICES_PER_AMOUNT_UNIT.items():
        print("\t", k, ":", v)
    print("-" * 80)
    print("CO2e emission prices (discounted):\n")
    for k, v in EMISSION_PRICE_SCENRIOS.items():
        print("\t", k, ":", v)
    print("-" * 80)
    print("Eskom equity (ZAR):", CURRENT_EQUITY)
    print("annual depreciation costs (ZAR):", CURRENT_DEPRECIATION_COSTS)
    print("annual maintenance costs (ZAR):", maintenance_costs)
    print("annual labour costs (ZAR):", labour_costs)
    print("annual sold energy (KWh):", CURRENT_YEARLY_SOLD_ENERGY)
    print("annual emissions (kg)):", eskom_co2e_emissions_2021_2022)
    print("-" * 80)
    print("Capacities (MW) and Energy mix:")
    for energy_type in ENERGY_TYPES:
        print("\t", energy_type, "-", "maximal:",
              CURRENT_ENERGY_MIX_MAX_CAPACITY[energy_type],
              CURRENT_ENERGY_MIX_MAX_RATIOS[energy_type],
              "realizable:",
              current_energy_mix_total_realizeable_capacities[energy_type],
              CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ[energy_type])
    print("\t", "total", "-", "maximal:",
          current_total_max_capacity,
          np.sum(list(CURRENT_ENERGY_MIX_MAX_RATIOS.values())),
          "realizable:",
          CURRENT_REALIZABLE_CAPACITY,
          np.sum(list(CURRENT_ENERGY_MIX_CAPACITY_FACTOR_ADJ.values())))
    print("-" * 80)
    print("power production factor:", power_production_factor)
    print("power production factor coal:", coal_power_production_factor)
    print("power production factor (pumped storage adjusted):",
          power_production_factor1)
    print("power production factor coal (pumped storage adjusted):",
          coal_power_production_factor1)
    print("emission factor:", EMISSIONS_FACTOR)
    print("depreciation factor:", DEPRECIATION_COST_FACTOR)
    print("-" * 80)
    print("capacity changes base case:\n")
    for k, v in IRP2030_CAPACITY_CHANGES.items():
        print("\t", k, ":")
        for v2 in v:
            print("\t\t", v2)
    print("-" * 80)
    print("capacity changes green continuation case:\n")
    for k, v in IRP2030_CONTINUED_CAPACITY_CHANGES.items():
        print("\t", k, ":")
        for v2 in v:
            print("\t\t", v2)
    print("-" * 80)
    print("capacity changes aggressive green continuation case:\n")
    for k, v in IRP2030_CONTINUED_CAPACITY_CHANGES_3.items():
        print("\t", k, ":")
        for v2 in v:
            print("\t\t", v2)
    print("-" * 80)











if __name__ == '__main__':
    pass

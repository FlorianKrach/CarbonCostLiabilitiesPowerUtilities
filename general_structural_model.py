"""
author: Florian Krach
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_config
import electricity_price_models
import os
import tqdm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import minimize
import time


# ------------------------------------------------------------------------------
# GLOBAL VARIABLES
TMAX = data_config.TMAX
ENERGY_TYPES_DESC = data_config.ENERGY_TYPES_DESC
FUEL_PRICES_PER_AMOUNT_UNIT = data_config.FUEL_PRICES_PER_AMOUNT_UNIT
ENERGY_TYPES = data_config.ENERGY_TYPES
# the following is meant as the emission reduction per time step per price unit
NET_EMISSION_REDUCTION_PER_PRICE_UNIT = \
    data_config.NET_EMISSION_REDUCTION_PER_PRICE_UNIT
NGFS_SCENARIO_NAMES_DICT = data_config.NGFS_SCENARIO_NAMES_DICT
ENERGY_TYPES_NAMES_DICT = data_config.ENERGY_TYPES_NAMES_DICT

# ------------------------------------------------------------------------------
# FUNCTIONS
def get_energy_mix_dict(energy_mix_list):
    d = {}
    for k, v in zip(ENERGY_TYPES, energy_mix_list):
        d[k] = v
    return d


def check_inputs(
        maturity, inv_discounting_factor, capacities, electricity_prices,
        energy_mix, emission_prices, NET_investments, investments,
        running_costs, inflation_factor):
    # check inputs
    if isinstance(inv_discounting_factor, float):
        inv_discounting_factor = inv_discounting_factor ** np.arange(1, maturity + 1)
    if isinstance(capacities, float):
        capacities = [capacities] * maturity
    if isinstance(electricity_prices, float):
        electricity_prices = [electricity_prices] * maturity
    if isinstance(energy_mix, dict):
        energy_mix = [energy_mix] * maturity
    if isinstance(emission_prices, float):
        emission_prices = [emission_prices] * maturity
    if isinstance(NET_investments, float):
        NET_investments = [NET_investments] * maturity
    if isinstance(inflation_factor, float):
        inflation_factor = \
            inflation_factor ** np.arange(1, maturity + 1)
    if isinstance(running_costs, float):
        running_costs = running_costs * \
                        np.array(inflation_factor)/np.array(inv_discounting_factor)
    if investments is None:
        investments = [0.] * maturity
    if isinstance(investments, float):
        investments = investments * \
                      np.array(inflation_factor)/np.array(inv_discounting_factor)
    return maturity, inv_discounting_factor, capacities, electricity_prices, \
           energy_mix, emission_prices, NET_investments, investments, \
           running_costs, inflation_factor


def compute_assets_liabilities_multipath(
        maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, electricity_price_paths, energy_mix, emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        start_time=0, investments=None,
        fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT, inflation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        **kwargs):
    """

    Args:
        maturity: int, number of time steps
        init_assets: float, initial assets value
        init_liabilities: float, initial liabilities value
        inv_discounting_factor: float or list of floats, discounting for future
            cash flows is done by division with this factor (-> the usual
            discountoung factor is 1/inv_discounting_factor). This factor
            corresponds to the bank account process.
        capacities: float or list of floats, maximal (i.e. not realizable!)
            capacities in capacity_units (e.g. KW) in each time step
        electricity_price_paths: np.array, dimension [nb_paths, maturity],
            the electricity prices in each time step
        energy_mix: dict or list of dicts, energy mix in each time step (of
            maximal capacities -> capacities are adjusted for capacity-factor
            and power-production-factor where needed)
            energy_mix[t][ENERGY_TYPES[i]] is the ratio of energy type i
            in time step t; sum of all energy_mix[t] has to be 1; also supports
            passing list instead of dict (in this case order has to be
            alphabetic, i.e., same as ENERGY_TYPES)
        emission_prices: float or list of floats, emission prices in each time
            step in price_unit/emission_unit, assumed to be inflation adjusted
            and discounted for today (-> no additional discounting done)
        NET_investments: float or list of floats, total investments in negative
            emissions technology in each time step in price_unit (not
            discounted)
        running_costs: float or list of floats, running costs in each time step
            (dicounted and inflation adjusted for today) in price_unit
        power_production_factor: list of dicts, the factor how much power is
            produced for each energy type from the theoretical realizable power
            output in each time step
        time_amount_per_step: float, the amount of time in each time step in
            the correct unit (e.g. hours if KWh is used)
        emissions_factor: float, the emissions factor by which the model
            computed emissions are multiplied to get the actual emissions
            calibrated to market data
        depreciation_factor: float, the depreciation factor by which the model
            computed depreciation is multiplied to get the actual depreciation
            calibrated to market data
        start_time: int, the time step at which the simulation starts
        investments: float or list of floats, investments in each time step in
            price_unit; are assumed to be discounted; these are any additional
            costs arising
        fuel_prices: None or dict of dicts, fuel prices in each time step,
            otherwise use FUEL_PRICES_PER_AMOUNT_UNIT; they are assumed to be
            inflation adjusted and discounted for today (-> no additional
            discounting done)
        inflation_factor: float or list, the inflation factor
        energy_types_desc: dict, description of the energy types
        NET_emission_reduction_per_price_unit: list of floats, the net emission
            reduction per price unit in each time step (in
            emission_unit/price_unit) is assumed to be inflation adjusted but
            not discounted

    - if lists are used they have to be at least of length T
    - use either proportional (by setting 'maintenance_cost_per_capacity_unit'
        in ENERGY_TYPES_DESC to value >0) or absolute (specified in
        'running_costs') maintenance costs, don't apply both at the same time
        (unless really wanted)
    - note that the algorithm treats investments and running costs the same way
    """
    energy_types = sorted(list(energy_types_desc.keys()))

    # check inputs
    maturity, inv_discounting_factor, capacities, _, \
    energy_mix, emission_prices, NET_investments, investments, running_costs, \
    inflation_factor = \
        check_inputs(
            maturity=maturity, inv_discounting_factor=inv_discounting_factor,
            capacities=capacities, electricity_prices=0.,
            energy_mix=energy_mix, emission_prices=emission_prices,
            NET_investments=NET_investments, investments=investments,
            running_costs=running_costs,
            inflation_factor=inflation_factor)

    # compute cash flows
    assets = np.zeros((len(electricity_price_paths), maturity+1-start_time))
    assets[:, 0] = init_assets
    liabilities = np.zeros((maturity+1-start_time,))
    liabilities[0] = init_liabilities
    for t in range(start_time, maturity):
        # compute cash flows
        energy_sales = 0
        pos_emissions = 0
        maintenance_costs = 0
        fuel_costs = 0
        depriciation_costs = 0
        for energy_type in energy_types:
            produced_energy = \
                energy_mix[t][energy_type] * capacities[t] * \
                energy_types_desc[energy_type]["capacity_factor"] *\
                power_production_factor[t][energy_type] * \
                time_amount_per_step
            energy_sales += \
                electricity_price_paths[:, t] * produced_energy / \
                inv_discounting_factor[t]
            pos_emissions += \
                produced_energy * emissions_factor * \
                energy_types_desc[energy_type]['emission_per_energy_unit']
            maintenance_costs += \
                energy_mix[t][energy_type] * capacities[t] * \
                energy_types_desc[energy_type][
                    "maintenance_cost_per_capacity_unit"] * \
                inflation_factor[t] / inv_discounting_factor[t]
            fuel_costs += \
                produced_energy * \
                energy_types_desc[energy_type]["amount_per_energy_unit"] * \
                fuel_prices[energy_type][t]
            depriciation_costs += \
                energy_mix[t][energy_type] * capacities[t] * \
                energy_types_desc[energy_type][
                    "yearly_depriciation_per_capacity_unit"] * \
                depreciation_factor * \
                inflation_factor[t] / inv_discounting_factor[t]
        neg_emissions = \
            NET_emission_reduction_per_price_unit[t]*NET_investments[t]
        emissions = pos_emissions - neg_emissions
        emission_costs = emission_prices[t]*emissions

        # update assets and liabilities
        assets[:, t+1] = energy_sales
        liabilities[t+1] = emission_costs + maintenance_costs + fuel_costs + \
                           investments[t] + running_costs[t] + \
                           depriciation_costs + \
                           NET_investments[t]/inv_discounting_factor[t]
        # if t==0:
        #     print("fuel costs (Billion ZAR):", fuel_costs/10**9)
        #     print("emission costs (Billion ZAR):", emission_costs/10**9)
        #     print("maintanance costs (Billion ZAR):", maintanance_costs/10**9)
        #     print("depriciation costs (Billion ZAR):", depriciation_costs/10**9)
        #     print("running costs (Billion ZAR):", running_costs[t]/10**9)
        #     print("total costs (Billion ZAR):", liabilities[t+1]/10**9)
        #     print("total revenue (mean) (Billion ZAR):", np.mean(assets[:, t+1])/10**9)
        #     return


    summed_assets = np.sum(assets, axis=1)
    summed_liabilities = np.sum(liabilities)

    return assets, liabilities, summed_assets, summed_liabilities


def compute_assets_liabilities(
        maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, electricity_prices, energy_mix, emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        start_time=0, investments=None,
        fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT, inflation_factor=1, **kwargs
):
    """
    Args:
        electricity_prices: float or list of floats, electricity prices in each
            time step in price_unit/capacity_unit

        other args see compute_assets_liabilities_multipath

        maturity:
        init_assets:
        init_liabilities:
        inv_discounting_factor:
        capacities:
        energy_mix:
        emission_prices:
        NET_investments:
        running_costs:
        power_production_factor:
        time_amount_per_step:
        emissions_factor:
        depreciation_factor:
        energy_types_desc:
        NET_emission_reduction_per_price_unit:
        start_time:
        investments:
        fuel_prices:
        inflation_factor:
        **kwargs:
    """
    electricity_prices = np.array(electricity_prices).reshape(1, -1)
    return compute_assets_liabilities_multipath(
        maturity=maturity, init_assets=init_assets,
        init_liabilities=init_liabilities,
        inv_discounting_factor=inv_discounting_factor, capacities=capacities,
        electricity_price_paths=electricity_prices, energy_mix=energy_mix,
        emission_prices=emission_prices, NET_investments=NET_investments,
        running_costs=running_costs, start_time=start_time,
        investments=investments, fuel_prices=fuel_prices,
        inflation_factor=inflation_factor, energy_types_desc=energy_types_desc,
        time_amount_per_step=time_amount_per_step,
        emissions_factor=emissions_factor,
        depreciation_factor=depreciation_factor,
        power_production_factor=power_production_factor,
        NET_emission_reduction_per_price_unit=
        NET_emission_reduction_per_price_unit,)


def get_min_electricity_price_to_stay_alive(
        maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, energy_mix, emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        init_capital_spending="no",
        investments=None, fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT,
        inflation_factor=1., **kwargs):
    """
    this function computes the minimum electricity price that is needed such
    that the company does not default (i.e. go bankrupt) at any time step.

    Args:
        init_capital_spending: str, one of {"no", "first", "equal"}
            "no": no capital spending to stay alive
            "first": full capital spending in first time step to stay alive
            "equal": capital spending in each time step is equal to
                (init_assets-init_liabilities)/maturity

        other args see compute_assets_liabilities

        maturity:
        init_assets:
        init_liabilities:
        inv_discounting_factor:
        capacities:
        energy_mix:
        emission_prices:
        NET_investments:
        running_costs:
        power_production_factor:
        time_amount_per_step:
        emissions_factor:
        depreciation_factor:
        energy_types_desc:
        NET_emission_reduction_per_price_unit:
        investments:
        fuel_prices:
        inflation_factor:
        **kwargs:
    """
    energy_types = sorted(list(energy_types_desc.keys()))

    # check inputs
    maturity, inv_discounting_factor, capacities, _, \
    energy_mix, emission_prices, NET_investments, investments, running_costs, \
    inflation_factor = \
        check_inputs(
            maturity=maturity, inv_discounting_factor=inv_discounting_factor,
            capacities=capacities, electricity_prices=0.,
            energy_mix=energy_mix, emission_prices=emission_prices,
            NET_investments=NET_investments, investments=investments,
            running_costs=running_costs, 
            inflation_factor=inflation_factor)

    # compute cash flows
    init_capital = init_assets - init_liabilities
    if init_capital_spending == "no":
        capital_spending = [0] * maturity
    elif init_capital_spending == "first":
        capital_spending = [init_capital] + [0] * (maturity-1)
    elif init_capital_spending == "equal":
        capital_spending = [init_capital/maturity] * maturity
    else:
        raise ValueError("init_capital_spending not supported")
    needed_electricity_prices = []

    for t in range(maturity):
        # compute cash flows
        pos_emissions = 0
        maintenance_costs = 0
        fuel_costs = 0
        depriciation_costs = 0
        total_energy_produced = 0
        for energy_type in energy_types:
            produced_energy = \
                energy_mix[t][energy_type] * capacities[t] * \
                energy_types_desc[energy_type]["capacity_factor"] *\
                power_production_factor[t][energy_type] * \
                time_amount_per_step
            total_energy_produced += produced_energy
            pos_emissions += \
                produced_energy * emissions_factor * \
                energy_types_desc[energy_type]['emission_per_energy_unit']
            maintenance_costs += \
                energy_mix[t][energy_type] * capacities[t] * \
                energy_types_desc[energy_type][
                    "maintenance_cost_per_capacity_unit"] * \
                inflation_factor[t] / inv_discounting_factor[t]
            fuel_costs += \
                produced_energy * \
                energy_types_desc[energy_type]["amount_per_energy_unit"] * \
                fuel_prices[energy_type][t]
            depriciation_costs += \
                energy_mix[t][energy_type] * capacities[t] * \
                energy_types_desc[energy_type][
                    "yearly_depriciation_per_capacity_unit"] * \
                depreciation_factor * \
                inflation_factor[t] / inv_discounting_factor[t]
        neg_emissions = \
            NET_emission_reduction_per_price_unit[t]*NET_investments[t]
        emissions = pos_emissions - neg_emissions
        emission_costs = emission_prices[t]*emissions

        # get electricity price, s.t. energy_sale + capital_spending = all costs
        elect_price = \
            (emission_costs + maintenance_costs + fuel_costs + investments[t] +
             running_costs[t] + NET_investments[t]/inv_discounting_factor[t] +
             depriciation_costs - capital_spending[t])/total_energy_produced * \
            inv_discounting_factor[t]
        needed_electricity_prices.append(elect_price)

    return needed_electricity_prices


def plot_min_electricity_prices(
        maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, energy_mix, scenarios_emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        investments=None, inflation_factor=1.,
        init_capital_spending="no",
        fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT,
        scenario_names=None, timegrid=None, inflation_adjusted=False,
        plot_postfix="", ax=None, **kwargs):
    """
    plots the minimum electricity price that is needed such that the company
    covers all costs (i.e. doesn't default)

    Args:
        plot_postfix: str, postfix for plot title
        inflation_adjusted: bool, if True, the electricity prices are adjusted

        other args see get_min_electricity_price_to_stay_alive

        maturity:
        init_assets:
        init_liabilities:
        inv_discounting_factor:
        capacities:
        energy_mix:
        scenarios_emission_prices:
        NET_investments:
        running_costs:
        power_production_factor:
        time_amount_per_step:
        emissions_factor:
        depreciation_factor:
        energy_types_desc:
        NET_emission_reduction_per_price_unit:
        investments:
        inflation_factor:
        init_capital_spending:
        fuel_prices:
        scenario_names:
        timegrid:
        ax: matplotlib axis object, possibility to plot on existing axis for
            compatibility with function multiplot
        **kwargs:
    """
    if isinstance(scenarios_emission_prices, dict):
        scenario_names = list(scenarios_emission_prices.keys())
        scenarios_emission_prices = list(scenarios_emission_prices.values())
    min_elec_prices = []
    for emission_prices in scenarios_emission_prices:
        min_elec_prices.append(get_min_electricity_price_to_stay_alive(
            maturity=maturity, init_assets=init_assets,
            init_liabilities=init_liabilities, fuel_prices=fuel_prices,
            inv_discounting_factor=inv_discounting_factor, capacities=capacities,
            energy_mix=energy_mix, emission_prices=emission_prices,
            NET_investments=NET_investments, running_costs=running_costs,
            investments=investments, inflation_factor=inflation_factor,
            init_capital_spending=init_capital_spending,
            power_production_factor=power_production_factor,
            time_amount_per_step=time_amount_per_step,
            emissions_factor=emissions_factor,
            depreciation_factor=depreciation_factor,
            energy_types_desc=energy_types_desc,
            NET_emission_reduction_per_price_unit=
            NET_emission_reduction_per_price_unit,))

    if scenario_names is None:
        scenario_names = ["scenario {}".format(i) for i in
                          range(len(scenarios_emission_prices))]
    if timegrid is None:
        timegrid = np.arange(maturity+1)
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(6.4, 4.8))
    else:
        ax1 = ax
    for i, min_elec_price in enumerate(min_elec_prices):
        # print("scenario {} -- running default prob: \n{}".format(
        #     scenario_names[i], running_default_prob))
        elec_prices = np.array(min_elec_price)
        if inflation_adjusted:
            elec_prices = elec_prices / np.array(inflation_factor)
        print("scenario {} -- inflation adjusted: {} -- "
              "min. electricity price: \n{}".format(
            scenario_names[i], inflation_adjusted, elec_prices))
        ax1.plot(timegrid[:-1], elec_prices,
                 label="{}".format(NGFS_SCENARIO_NAMES_DICT[scenario_names[i]]),
                 color="C{}".format(i), ls="-")
    ax1.set_xlabel('time')
    ax1.set_ylabel('minimum electricity price [ZAR/KWh]')
    title = "Min. electricity price to cover all costs"
    if inflation_adjusted:
        title += " (inflation adjusted)"
    if not os.path.exists("results"):
        os.mkdir("results")
    fname = "results/min_electricity_prices_costcover.pdf"
    if ax is None:
        plt.title(title)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
        plt.savefig(
            "results/min_electricity_prices_costcover_{}.pdf".format(
                plot_postfix), bbox_inches="tight")
        plt.close()
    return fname, title, [None]


def print_values_from_paper():
    data_config.print_values_from_paper()


def compute_default_probability(
        maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, electricity_price_paths, energy_mix, emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        investments=None, inflation_factor=1.,
        use_running_default_probs=False, fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT,
        **kwargs):
    """
    this function computes the MC-approximation of the default probability of an
    electricity generating company for given (sampled) electricity price paths.

    Args:
        use_running_default_probs: bool, whether to compute running default
            probability or the instentaneous probability of default

        other args see compute_assets_liabilities_multipath

        maturity:
        init_assets:
        init_liabilities:
        inv_discounting_factor:
        capacities:
        electricity_price_paths:
        energy_mix:
        emission_prices:
        NET_investments:
        running_costs:
        power_production_factor:
        time_amount_per_step:
        emissions_factor:
        depreciation_factor:
        energy_types_desc:
        NET_emission_reduction_per_price_unit:
        investments:
        inflation_factor:
        fuel_prices:
        **kwargs:
    """
    nb_paths = len(electricity_price_paths)
    assets, liabilities, summed_assets, summed_liabilities = \
        compute_assets_liabilities_multipath(
            maturity=maturity, init_assets=init_assets,
            init_liabilities=init_liabilities,
            inv_discounting_factor=inv_discounting_factor,
            capacities=capacities,
            electricity_price_paths=electricity_price_paths,
            energy_mix=energy_mix, emission_prices=emission_prices,
            NET_investments=NET_investments, investments=investments,
            running_costs=running_costs, fuel_prices=fuel_prices,
            inflation_factor=inflation_factor,
            power_production_factor=power_production_factor,
            time_amount_per_step=time_amount_per_step,
            emissions_factor=emissions_factor,
            depreciation_factor=depreciation_factor,
            energy_types_desc=energy_types_desc,
            NET_emission_reduction_per_price_unit=
            NET_emission_reduction_per_price_unit)
    cumsum_assets = np.cumsum(assets, axis=1)
    cumsum_liabilities = np.cumsum(liabilities).reshape((1, -1)).repeat(
        nb_paths, axis=0)
    if not use_running_default_probs:
        summed_liabilities = np.array([summed_liabilities] * nb_paths)
        defaults = summed_assets < summed_liabilities
        defaults_each_maturity = cumsum_assets < cumsum_liabilities
        default_prob_all_maturities = np.mean(defaults_each_maturity, axis=0)
        default_prob_all_maturities_std = np.std(defaults_each_maturity, axis=0)
        return np.mean(defaults), np.std(defaults), \
               default_prob_all_maturities, default_prob_all_maturities_std
    else:
        defaults = np.any(
            cumsum_assets < cumsum_liabilities, axis=1)
        default_prob, default_prob_std = np.mean(defaults), np.std(defaults)
        default_until_t = np.minimum(
            np.cumsum(cumsum_assets<cumsum_liabilities, axis=1), 1)
        running_default_prob = np.mean(default_until_t, axis=0)
        running_default_prob_std = np.std(default_until_t, axis=0)
        return default_prob, default_prob_std, running_default_prob,\
               running_default_prob_std


def plot_running_default_prob_and_scenarios(
        maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, energy_mix, scenarios_emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        investments=None,
        inflation_factor=1., fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT,
        nb_paths=10000, seed=0, init_electricity_price=1.,
        mean_percentage_jump=0.1, scenario_names=None, timegrid=None,
        use_running_default_probs=True,
        plot_scenarios_only=False,
        plot_postfix="", default_prob_credit_ratings=None,
        electricity_price_scaling_factor=None, ax=None,
        plot_default_probs_only=False,
        **kwargs):
    """

    Args:
        scenarios_emission_prices: dict or list, the different CO2e price
            scenarios, if dict, then the keys are the scenario names
        mean_percentage_jump: float, the mean percentage jump lambda of the
            electricity price for the electricity price model
        nb_paths: int, number of paths to sample from electricity price model
            used in MC approximation
        seed: int, seed for random number generator
        init_electricity_price: float, initial electricity price
        electricity_price_scaling_factor: None or list of floats, the scaling
            factors for the electricity price, if None, then no scaling otherwise
            the length of the list must be the same as maturity
        scenario_names: None or list of strings, provide for names in plot
            legend
        timegrid: None or np.array, if None, then np.arange(maturity), otherwise
            the timestamps for plotting
        use_running_default_probs: bool, if True, then the running default
            probabilities are plotted (i.e. default anywhen up to maturity),
            otherwise the instantaneous default probabilities are plotted for
            different maturities (i.e. at each time the probability that it
            defaults there)
        plot_scenarios_only: bool, if True, then only the scenarios are plotted
        plot_postfix: str, postfix for plot file name
        default_prob_credit_ratings: None or dict, if None, then nothing plotted
            otherwise plot the default probabilities for the different credit
            ratings
        ax: matplotlib axis object, possibility to plot on existing axis for
            compatibility with function multiplot
        plot_default_probs_only: bool, if True, then only the default probs are
            plotted

        other args see compute_default_probability

        maturity:
        init_assets:
        init_liabilities:
        inv_discounting_factor:
        capacities:
        energy_mix:
        NET_investments:
        running_costs:
        power_production_factor:
        time_amount_per_step:
        emissions_factor:
        depreciation_factor:
        energy_types_desc:
        NET_emission_reduction_per_price_unit:
        investments:
        inflation_factor:
        fuel_prices:
        **kwargs:
    """
    if isinstance(scenarios_emission_prices, dict):
        scenario_names = list(scenarios_emission_prices.keys())
        scenarios_emission_prices = list(scenarios_emission_prices.values())
    running_default_probs = []
    for emission_prices in scenarios_emission_prices:
        np.random.seed(seed)
        electricity_price_paths = \
            electricity_price_models.sample_exponential_percentage_price_jumps_paths(
                mean_percentage=mean_percentage_jump, T=maturity,
                initial_price=init_electricity_price, nb_samples=nb_paths,
                factors=electricity_price_scaling_factor)
        _, _, running_default_prob, running_default_prob_std = \
            compute_default_probability(
                maturity=maturity, init_assets=init_assets,
                init_liabilities=init_liabilities,
                inv_discounting_factor=inv_discounting_factor,
                capacities=capacities,
                electricity_price_paths=electricity_price_paths,
                energy_mix=energy_mix,
                emission_prices=emission_prices,
                NET_investments=NET_investments, investments=investments,
                inflation_factor=inflation_factor,
                running_costs=running_costs, fuel_prices=fuel_prices,
                use_running_default_probs=use_running_default_probs,
                power_production_factor=power_production_factor,
                time_amount_per_step=time_amount_per_step,
                emissions_factor=emissions_factor,
                depreciation_factor=depreciation_factor,
                energy_types_desc=energy_types_desc,
                NET_emission_reduction_per_price_unit=
                NET_emission_reduction_per_price_unit,)
        running_default_probs.append(running_default_prob)

    if scenario_names is None:
        scenario_names = ["scenario {}".format(i) for i in
                          range(len(scenarios_emission_prices))]
    if timegrid is None:
        timegrid = np.arange(maturity+1)
    if ax is None:
        fig, ax1 = plt.subplots()
        if not plot_scenarios_only:
            ax2 = ax1.twinx()
    else:
        ax1, ax2 = ax
    if not plot_default_probs_only:
        ax_default_probs = ax2
    else:
        ax_default_probs = ax1
    if default_prob_credit_ratings is not None and not plot_scenarios_only:
        for credit_rating, default_prob in default_prob_credit_ratings.items():
            ax_default_probs.plot(
                timegrid[:-1], default_prob[:len(timegrid)-1],
                label="{}".format(credit_rating),
                color="gray", ls="-", alpha=0.75)
            ax_default_probs.text(
                timegrid[-2]+0.5, default_prob[:len(timegrid)][-2]-0.01,
                "{}".format(credit_rating), color="gray", alpha=0.75)
    for i, running_default_prob in enumerate(running_default_probs):
        # print("scenario {} -- running default prob: \n{}".format(
        #     scenario_names[i], running_default_prob))
        if not plot_default_probs_only:
            ax1.plot(timegrid[:-1],
                     scenarios_emission_prices[i]*inv_discounting_factor,
                     label="{}".format(NGFS_SCENARIO_NAMES_DICT[scenario_names[i]]),
                     color="C{}".format(i), ls="-")
        if not plot_scenarios_only:
            ax_default_probs.plot(
                timegrid, running_default_prob,
                label="{}".format(
                    NGFS_SCENARIO_NAMES_DICT[scenario_names[i]]),
                color="C{}".format(i), ls="--")
    ax1.set_xlabel('time')
    if not plot_default_probs_only:
        ax1.set_ylabel('$CO_2e$ price $[ZAR/kg]$')
    legend_elements = None
    labels = None
    if not plot_scenarios_only:
        y2label = "Probability of default" if use_running_default_probs \
            else "Instantaneous probability of default"
        ax_default_probs.set_ylabel(y2label)
        if ax is None:
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

        legend_elements = [
            Line2D([0], [0], color='black', label='$CO_2e$ price'),
            Line2D([0], [0], color='black', ls="--", label=y2label),]
        if plot_default_probs_only:
            legend_elements = legend_elements[1:]
        labels = [l.get_label() for l in legend_elements]
        if ax is None:
            ax2.legend(handles=legend_elements, loc='lower center',
                       bbox_to_anchor=(0.5, 1.1), ncol=2)
        # ax2.legend(loc='upper left', bbox_to_anchor=(1.2, 0.49))
        title_ = y2label
        if not plot_default_probs_only:
            title_ += " and $CO_2e$ price"
        ax1.set_title(title_)
    else:
        if ax is None:
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax1.set_title("$CO_2e$ price")
    if not os.path.exists("results"):
        os.mkdir("results")
    if use_running_default_probs:
        figname = "results/running_default_prob_and_co2e_price_" \
                  "{}.pdf"
    else:
        figname = "results/default_prob_and_co2e_price_{}.pdf"
    if plot_scenarios_only:
        figname = "results/NGFS_scenarios.pdf"
    if ax is None:
        plt.savefig(figname.format(plot_postfix), bbox_inches="tight")
        plt.close()
    return None, None, [None, [legend_elements, labels]]



def compute_bond_price(
        maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, electricity_price_paths, energy_mix, emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        investments=None, inflation_factor=1.,
        fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT,
        use_running_default_probs=True, **kwargs):
    """
    this function computes the MC-approximation of the (defaultable) bond price
    term structure of an electricity generating company for given (sampled)
    electricity price paths.

    uses same parameters as compute_default_probability
    """
    _, _, default_probs, _ = compute_default_probability(
        maturity=maturity, init_assets=init_assets,
        init_liabilities=init_liabilities,
        inv_discounting_factor=inv_discounting_factor, capacities=capacities,
        electricity_price_paths=electricity_price_paths, energy_mix=energy_mix,
        emission_prices=emission_prices, NET_investments=NET_investments,
        running_costs=running_costs, investments=investments,
        inflation_factor=inflation_factor, fuel_prices=fuel_prices,
        use_running_default_probs=use_running_default_probs,
        power_production_factor=power_production_factor,
        time_amount_per_step=time_amount_per_step,
        emissions_factor=emissions_factor,
        depreciation_factor=depreciation_factor,
        energy_types_desc=energy_types_desc,
        NET_emission_reduction_per_price_unit=
        NET_emission_reduction_per_price_unit,)
    bond_price = (1-default_probs)/np.array([1.]+list(inv_discounting_factor))

    return bond_price


def plot_bond_price_term_structure(
        max_maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, energy_mix, scenarios_emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        investments=None, inflation_factor=1.,
        fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT,
        nb_paths=10000, seed=0, init_electricity_price=1.,
        mean_percentage_jump=0.1, scenario_names=None, timegrid=None,
        use_running_default_probs=True, plot_postfix="",
        electricity_price_scaling_factor=None, ax=None,
        plot_default_probs_only=False,
        **kwargs):
    """
    this function plots the term structure of the bond price for different
    scenarios of the $CO_2e$ price.

    same parameters as plot_running_default_prob_and_scenarios
    """
    if isinstance(scenarios_emission_prices, dict):
        scenario_names = list(scenarios_emission_prices.keys())
        scenarios_emission_prices = list(scenarios_emission_prices.values())
    bond_prices = []
    for emission_prices in scenarios_emission_prices:
        np.random.seed(seed)
        electricity_price_paths = \
            electricity_price_models.sample_exponential_percentage_price_jumps_paths(
                mean_percentage=mean_percentage_jump, T=max_maturity,
                initial_price=init_electricity_price, nb_samples=nb_paths,
                factors=electricity_price_scaling_factor)
        bond_price = compute_bond_price(
            maturity=max_maturity, init_assets=init_assets,
            init_liabilities=init_liabilities,
            inv_discounting_factor=inv_discounting_factor,
            capacities=capacities,
            electricity_price_paths=electricity_price_paths,
            energy_mix=energy_mix,
            emission_prices=emission_prices,
            NET_investments=NET_investments, investments=investments,
            inflation_factor=inflation_factor,
            running_costs=running_costs, fuel_prices=fuel_prices,
            use_running_default_probs=use_running_default_probs,
            power_production_factor=power_production_factor,
            time_amount_per_step=time_amount_per_step,
            emissions_factor=emissions_factor,
            depreciation_factor=depreciation_factor,
            energy_types_desc=energy_types_desc,
            NET_emission_reduction_per_price_unit=
            NET_emission_reduction_per_price_unit,)
        bond_prices.append(bond_price)

    if scenario_names is None:
        scenario_names = ["scenario {}".format(i) for i in
                          range(len(scenarios_emission_prices))]
    if timegrid is None:
        timegrid = np.arange(max_maturity + 1)
    if ax is None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
    else:
        ax1, ax2 = ax
    if not plot_default_probs_only:
        ax_default_probs = ax2
    else:
        ax_default_probs = ax1
    for i, bond_price in enumerate(bond_prices):
        # print("scenario {} -- running default prob: \n{}".format(
        #     scenario_names[i], running_default_prob))
        if not plot_default_probs_only:
            ax1.plot(
                timegrid[:-1],
                scenarios_emission_prices[i]*inv_discounting_factor,
                label="{}".format(NGFS_SCENARIO_NAMES_DICT[scenario_names[i]]),
                color="C{}".format(i), ls="-")
        ax_default_probs.plot(
            timegrid, bond_price, label="{}".format(
                NGFS_SCENARIO_NAMES_DICT[scenario_names[i]]),
            color="C{}".format(i), ls="--")
    ax1.set_xlabel('time')
    if not plot_default_probs_only:
        ax1.set_ylabel('$CO_2e$ price $[ZAR/kg]$')
    ax_default_probs.set_ylabel('bond price')
    legend_elements = [
        Line2D([0], [0], color='black', label='$CO_2e$ price'),
        Line2D([0], [0], color='black', ls="--", label="bond price"), ]
    if plot_default_probs_only:
        legend_elements = legend_elements[1:]
    labels = [l.get_label() for l in legend_elements]
    title_ = "Bond price"
    if not plot_default_probs_only:
        title_ += " and $CO_2e$ price"
    title_ += " for different maturities"
    ax1.set_title(title_)
    if not os.path.exists("results"):
        os.mkdir("results")
    if ax is None:
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax2.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 1.1), ncol=2)
        plt.savefig(
            "results/bond_price_and_co2e_price_{}.pdf".format(plot_postfix),
            bbox_inches="tight")
        plt.close()
    return None, None, [None, [legend_elements, labels]]



def plot_power_plant_cost_comparison(
        maturity, fuel_prices, scenarios_emission_prices,
        inflation_factor, inv_discounting_factor,
        time_amount_per_step=365*24.,
        energy_types_desc=ENERGY_TYPES_DESC,
        timegrid=None,
        scenario_names=None, plots_per_line=4, splits=1,
        cost_comp_figsize=(8, 6), us_inv_discounting_factor=None,
        **kwargs):
    """
    this function plots the cost for each power plant type in different
    scenarios of the $CO_2e$ price.

    Args:
        maturity:
        fuel_prices:
        scenarios_emission_prices:
        inflation_factor:
        inv_discounting_factor:
        timegrid:
        scenario_names:
        plots_per_line: number of plots per line in the figure
        cost_comp_figsize: figure size
        us_inv_discounting_factor:

    Returns:

    """
    energy_types = sorted(list(energy_types_desc.keys()))

    if isinstance(scenarios_emission_prices, dict):
        scenario_names = list(scenarios_emission_prices.keys())
        scenarios_emission_prices = list(scenarios_emission_prices.values())
    if timegrid is None:
        timegrid = np.arange(maturity + 1)
    if scenario_names is None:
        scenario_names = ["scenario {}".format(i) for i in
                          range(len(scenarios_emission_prices))]
    splits_scenarios_emission_prices = np.split(
        np.array(scenarios_emission_prices), splits)
    scenario_names = np.split(np.array(scenario_names), splits)
    for k, scenarios_emission_prices in enumerate(
            splits_scenarios_emission_prices):
        fig, axs = plt.subplots(
            (len(scenarios_emission_prices.tolist()) + plots_per_line-1) //
            plots_per_line,
            plots_per_line, figsize=cost_comp_figsize)
        fig.tight_layout()
        for j, emission_prices in enumerate(scenarios_emission_prices):
            curr_ax = axs[j // plots_per_line, j % plots_per_line]
            costs = []
            for t in range(maturity):
                costs_t = []
                for energy_type in energy_types:
                    emission_costs = \
                        energy_types_desc[energy_type][
                            'emission_per_energy_unit'] * emission_prices[t] * \
                        inv_discounting_factor[t] / inflation_factor[t]
                    fuel_costs = \
                        energy_types_desc[energy_type][
                                     "amount_per_energy_unit"] * \
                        fuel_prices[energy_type][t] * \
                        inv_discounting_factor[t] / inflation_factor[t]
                    depriciation_costs = \
                        energy_types_desc[energy_type][
                            "yearly_depriciation_per_capacity_unit"] / \
                        energy_types_desc[energy_type]["capacity_factor"] / \
                        time_amount_per_step
                    total_costs = \
                        emission_costs + fuel_costs + depriciation_costs
                    costs_t.append(total_costs)
                costs.append(costs_t)
            costs = np.array(costs)

            for i, energy_type in enumerate(energy_types):
                curr_ax.plot(
                    timegrid[:-1], costs[:, i],
                    label=ENERGY_TYPES_NAMES_DICT[energy_type],
                    color="C{}".format(i))
            curr_ax.set_title("{}".format(
                NGFS_SCENARIO_NAMES_DICT[scenario_names[k][j]]))
        for ax in axs.flat:
            ax.set(xlabel='time', ylabel='costs [ZAR/KWh]')
        for ax in axs.flat:
            ax.label_outer()
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5,
                   bbox_to_anchor=(0.5, 1.005))
        fig.suptitle("Inflation Adjusted Costs (depriciation + fuel + "
                     "$CO_2e$-emission) per KWh",
                     y=1.15, fontsize=16)
        if not os.path.exists("results"):
            os.mkdir("results")
        plt.savefig(
            "results/costs_per_energy_type_scenario_{}.pdf".format(k),
            bbox_inches="tight")
        plt.close()


def plot_mean_electricity_prices_for_MPPJ(
        maturity, init_electricity_price, mean_percentages_to_plot,
        nb_paths, timegrid=None, inflation_factor=None, inflation_adjusted=True,
        seed=0, plot_mppj_std=False, plot_postfix=None,
        electricity_price_scaling_factor=None, ax=None, **kwargs):
    """
    this function plots the mean electricity price for different mean
    percentage price jump values.

    Args:
        maturity:
        init_electricity_price:
        mean_percentages_to_plot: list of mean percentage price jump values for
            which mean electricity prices should be plotted
        initial_price:
        nb_paths:
        timegrid:
        inflation_factor:
        inflation_adjusted:
        electricity_price_scaling_factor:
        seed:
        ax: matplotlib axis object, possibility to plot on existing axis for
            compatibility with function multiplot

    """
    if timegrid is None:
        timegrid = np.arange(maturity+1)
    means = []
    stds = []
    for MPPJ in mean_percentages_to_plot:
        np.random.seed(seed)
        electricity_price_paths = \
            electricity_price_models.sample_exponential_percentage_price_jumps_paths(
                mean_percentage=MPPJ, T=maturity,
                initial_price=init_electricity_price, nb_samples=nb_paths,
                factors=electricity_price_scaling_factor)
        if inflation_adjusted and inflation_factor is not None:
            electricity_price_paths = electricity_price_paths / \
                inflation_factor.reshape(1, -1).repeat(nb_paths, axis=0)
        else:
            inflation_adjusted = False
        means.append(np.mean(electricity_price_paths, axis=0))
        stds.append(np.std(electricity_price_paths, axis=0))

    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
    for i, MPPJ in enumerate(mean_percentages_to_plot):
        ax1.plot(timegrid[:-1], means[i], label="MPPJ={}".format(MPPJ),
                color="C{}".format(i))
        if plot_mppj_std:
            ax1.fill_between(
                timegrid[:-1], means[i] - stds[i], means[i] + stds[i],
                color="C{}".format(i), alpha=0.2)
    ax1.set_xlabel("time")
    ax1.set_ylabel("electricity price [ZAR/KWh]")
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax1.legend()
    title = "Mean electricity price"
    if inflation_adjusted:
        title += " (inflation adjusted)"
    ax1.set_title(title)
    if not os.path.exists("results"):
        os.mkdir("results")
    if plot_postfix is None:
        plot_postfix = ""
    if ax is None:
        plt.savefig(
            "results/mean_electricity_price_for_different_MPPJ_{}"
            ".pdf".format(plot_postfix),
            bbox_inches="tight")
        plt.close()
    return None, None, [None]


def plot_capacities_and_energy_mix(
        capacities, energy_mix, timegrid=None,
        power_production_factor=None,
        which_cap_type_to_plot="maximal",
        energy_types_desc=ENERGY_TYPES_DESC,
        plot_postfix=None, ax=None, **kwargs):
    """
    this function plots the capacities and energy mix over time

    Args:
        see also plot_running_default_prob_and_scenarios for argument
        specification

        capacities: assumed to be maximal capacities
        energy_mix: assumed to be ratios of maximal capacities
        timegrid:
        power_production_factor:
        energy_types_desc:
        which_cap_type_to_plot: one of {"maximal", "realizable", "running"}
        plot_postfix:
        ax: matplotlib axis object, possibility to plot on existing axis for
            compatibility with function multiplot
        **kwargs:

    """
    if timegrid is None:
        timegrid = np.arange(len(capacities))
    energy_types = sorted(list(energy_mix[0].keys()))
    energy_type_caps = []
    for t in range(len(capacities)):
        cap = []
        for energy_type in energy_types:
            c_of_e_type = capacities[t]*energy_mix[t][energy_type]
            if which_cap_type_to_plot == "running":
                c_of_e_type *= power_production_factor[t][energy_type] * \
                               energy_types_desc[energy_type]["capacity_factor"]
            if which_cap_type_to_plot == "realizable":
                c_of_e_type *= energy_types_desc[energy_type]["capacity_factor"]
            cap.append(c_of_e_type)
        energy_type_caps.append(np.array(cap))
    energy_type_caps = np.array(energy_type_caps)

    if ax is None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
    else:
        ax1, ax2 = ax
    summed_caps = np.sum(energy_type_caps, axis=1)
    ax1.plot(
        timegrid[:-1], summed_caps, color="black", ls="--",
        label="total {} capacity".format(which_cap_type_to_plot))
    for i, energy_type in enumerate(energy_types):
        ax2.plot(timegrid[:-1], energy_type_caps[:, i]/summed_caps,
                 label="{}".format(ENERGY_TYPES_NAMES_DICT[energy_type]),
                 color="C{}".format(i), ls="-")

    ax1.set_xlabel('time')
    ax1.set_ylabel('capacity [KW]')
    ax2.set_ylabel('proportion of energy mix')
    if not os.path.exists("results"):
        os.mkdir("results")
    title = "{} capacities & energy mix".format(
        which_cap_type_to_plot.capitalize())
    if ax is None:
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
        plt.title(title)
        plt.savefig(
            "results/capacities_energy_mix_{}.pdf".format(plot_postfix),
            bbox_inches="tight")
        plt.close()
    fname = "results/capacities_energy_mix.pdf"
    return fname, title, [None, None]



def fit_electricity_price_scaling_factors(
        maturity, init_assets, init_liabilities, inv_discounting_factor,
        capacities, energy_mix, emission_prices,
        NET_investments, running_costs,
        power_production_factor,
        time_amount_per_step=365*24., emissions_factor=1.,
        depreciation_factor=1.,
        energy_types_desc=ENERGY_TYPES_DESC,
        NET_emission_reduction_per_price_unit=
        NET_EMISSION_REDUCTION_PER_PRICE_UNIT,
        investments=None,
        fuel_prices=FUEL_PRICES_PER_AMOUNT_UNIT, inflation_factor=1.,
        use_running_default_probs=True,
        default_probs=None, nb_paths=10000, seed=0,
        init_electricity_price=1.,
        mean_percentage_jump=0.1,
        plot_postfix="", timegrid=None,
        electricity_price_scaling_factor=None,
        first_fixed=True, fit_mppj=False, fit_init_assets=False,
        optimization_kwargs=None,
        ax=None, **kwargs):
    """
    this function fits the electricity price scaling factor to the market data
    of probabilities of default

    Args:
        electricity_price_scaling_factor: starting value for the fit
        first_fixed: if True, the first value of the scaling factor is fixed
            to be 1
        fit_mppj: if True, the mean percentage price jump is fitted as well
        fit_init_assets: if True, the initial assets are fitted as well
        optimization_kwargs: kwargs passed to the minimization function;
            should at least include "method"

        other args see get_exp_perc_price_jump_param_for_default_probability

        maturity:
        init_assets:
        init_liabilities:
        inv_discounting_factor:
        capacities:
        energy_mix:
        emission_prices:
        NET_investments:
        running_costs:
        power_production_factor:
        time_amount_per_step:
        emissions_factor:
        depreciation_factor:
        energy_types_desc:
        NET_emission_reduction_per_price_unit:
        investments:
        fuel_prices:
        inflation_factor:
        use_running_default_probs:
        default_probs:
        nb_paths:
        seed:
        init_electricity_price:
        mean_percentage_jump:
        plot_postfix:
        timegrid:
        ax: matplotlib axis object, possibility to plot on existing axis for
            compatibility with function multiplot
    """

    def opt_func(
            factors, first_fixed=True, fit_mppj=False, fit_init_assets=False):
        if fit_mppj:
            mppj_param = factors[0]
            factors = factors[1:]
        else:
            mppj_param = mean_percentage_jump
        if fit_init_assets:
            init_assets_param = factors[0]
            factors = factors[1:]
        else:
            init_assets_param = 1.
        if first_fixed:
            _factors = np.concatenate([[1.], factors])
        else:
            _factors = factors
        np.random.seed(seed)
        electricity_price_paths = \
            electricity_price_models.sample_exponential_percentage_price_jumps_paths(
                mean_percentage=mppj_param, T=maturity,
                initial_price=init_electricity_price, nb_samples=nb_paths,
                factors=_factors)
        _, _, default_prob_all_maturities, std = \
            compute_default_probability(
                maturity=maturity, init_assets=init_assets*init_assets_param,
                init_liabilities=init_liabilities,
                inv_discounting_factor=inv_discounting_factor,
                capacities=capacities,
                electricity_price_paths=electricity_price_paths,
                energy_mix=energy_mix, emission_prices=emission_prices,
                NET_investments=NET_investments, investments=investments,
                running_costs=running_costs,
                use_running_default_probs=use_running_default_probs,
                fuel_prices=fuel_prices, inflation_factor=inflation_factor,
                time_amount_per_step=time_amount_per_step,
                emissions_factor=emissions_factor,
                depreciation_factor=depreciation_factor,
                power_production_factor=power_production_factor,
                energy_types_desc=energy_types_desc,
                NET_emission_reduction_per_price_unit=
                NET_emission_reduction_per_price_unit, )
        diff = np.sum((default_prob_all_maturities - default_probs)**2)
        return diff

    if isinstance(fit_init_assets, float):
        init_assets_param = fit_init_assets
        fit_init_assets = True
    else:
        init_assets_param = 1.
    wrapper_opt_func = \
        lambda factors: opt_func(
            factors, first_fixed=first_fixed, fit_mppj=fit_mppj,
            fit_init_assets=fit_init_assets)
    if first_fixed:
        x0 = electricity_price_scaling_factor[1:]
    else:
        x0 = electricity_price_scaling_factor
    if fit_init_assets:
        x0 = np.concatenate([[init_assets_param], x0])
    if fit_mppj:
        x0 = np.concatenate([[mean_percentage_jump], x0])
    t = time.time()
    res = minimize(wrapper_opt_func, x0=x0, **optimization_kwargs)
    opt_time = time.time() - t
    optimal_factors = res.x
    mppj_param = mean_percentage_jump
    if fit_mppj:
        mppj_param = optimal_factors[0]
        optimal_factors = optimal_factors[1:]
    init_assets_param = 1.
    if fit_init_assets:
        init_assets_param = optimal_factors[0]
        optimal_factors = optimal_factors[1:]
    if first_fixed:
        optimal_factors = np.concatenate([[1.], optimal_factors])
    # print("optimization result:\n", res)
    print("-"*80)
    print("optimal factors:", optimal_factors)
    if fit_mppj:
        print("optimal mean percentage jump:", mppj_param)
    if fit_init_assets:
        print("optimal init assets param:", init_assets_param)
    print("diff with optimal factors", wrapper_opt_func(res.x))
    print("time for optimization:", opt_time)
    print("-" * 80)

    # compute default probs with optimal factors
    np.random.seed(seed)
    electricity_price_paths = \
        electricity_price_models.sample_exponential_percentage_price_jumps_paths(
            mean_percentage=mppj_param, T=maturity,
            initial_price=init_electricity_price, nb_samples=nb_paths,
            factors=optimal_factors)
    _, _, default_prob_all_maturities, std = \
        compute_default_probability(
            maturity=maturity, init_assets=init_assets*init_assets_param,
            init_liabilities=init_liabilities,
            inv_discounting_factor=inv_discounting_factor,
            capacities=capacities,
            electricity_price_paths=electricity_price_paths,
            energy_mix=energy_mix, emission_prices=emission_prices,
            NET_investments=NET_investments, investments=investments,
            running_costs=running_costs,
            use_running_default_probs=use_running_default_probs,
            fuel_prices=fuel_prices, inflation_factor=inflation_factor,
            time_amount_per_step=time_amount_per_step,
            emissions_factor=emissions_factor,
            depreciation_factor=depreciation_factor,
            power_production_factor=power_production_factor,
            energy_types_desc=energy_types_desc,
            NET_emission_reduction_per_price_unit=
            NET_emission_reduction_per_price_unit, )

    # plot market default probability curve and calibrated default prob curve
    if timegrid is None:
        timegrid = np.arange(maturity + 1)
    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
    ax1.plot(timegrid, default_probs, label="market")
    ax1.plot(timegrid, default_prob_all_maturities,
             label="calibrated model", color="C1")
    # ci = np.sqrt(computed_default_probs_all_maturities[idx] -
    #               computed_default_probs_all_maturities[idx]**2) / \
    #      np.sqrt(nb_paths) * 1.96
    # ax1.fill_between(
    #     timegrid,
    #     computed_default_probs_all_maturities[idx] - ci,
    #     computed_default_probs_all_maturities[idx] + ci,
    #     alpha=0.3, color="C1")
    ax1.set_xlabel("maturity")
    ax1.set_ylabel("default probability")
    ax1.legend()
    ax1.set_title("Term structure of probability of default")
    if not os.path.exists("results"):
        os.mkdir("results")
    if ax is None:
        plt.savefig(
            "results/calibrated_elect_factors_model_default_prob_vs_market_{}"
            ".pdf".format(plot_postfix), bbox_inches='tight')
        plt.close()
    return None, None, [None]



def multiplot(
        plotfunc, configs, plots_per_line=3, figsize=(6.4, 4.8),
        suptitle_kwargs=dict(y=1.15, fontsize=16),
        legend_kwargs=dict(loc='lower center', ncol=5,
                           bbox_to_anchor=(0.5, 1.005)),
        titles=None, twinx=False,
        twinx_legend_kwargs=None,
        plot_suptitle=True,
        remove_unnecessary_xlabel=True,
        remove_unnecessary_ylabel=True,
        remove_unnecessary_twinxlabel=True,
        remove_unnecessary_twinxticks=True,
        remove_unnecessary_yticks=False,
        legend_outside=True,
        fname_set=None,
        plot_legend=[True, True],
):
    """
    Plot multiple plots in one figure.

    Args:
        plotfunc: str or function, the function to call for each plot
        configs: list of dicts, the configs for each plot
        plots_per_line: int, number of plots per line
        figsize: tuple, figure size
        suptitle_kwargs: dict, kwargs for suptitle
        legend_kwargs: dict, kwargs for legend
        titles: None or list of str, titles for each plot
        twinx: bool, whether to use twinx for each plot
        twinx_legend_kwargs: None or dict, kwargs for twinx legend
    """

    suptitle = None
    fname = None
    legend_handles = [None]
    if not isinstance(plotfunc, list):
        plotfunc = [plotfunc] * len(configs)
    for i, _plotfunc in enumerate(plotfunc):
        if isinstance(_plotfunc, str):
            plotfunc[i] = eval(_plotfunc)
    nb_plots = len(configs)
    nb_lines = int(np.ceil(nb_plots / plots_per_line))
    fig, axs = plt.subplots(nb_lines, plots_per_line, figsize=figsize)
    twinx_axis = []
    for i, config in enumerate(configs):
        if nb_lines == 1:
            ax = axs[i]
        else:
            ax = axs[i // plots_per_line, i % plots_per_line]
        if twinx:
            ax1 = ax
            ax2 = ax1.twinx()
            twinx_axis.append(ax2)
            ax = (ax1, ax2)
        fname, suptitle, legend_handles = plotfunc[i](ax=ax, **config)
    for i, ax in enumerate(axs.flat):
        # ax.label_outer()
        line, col = i // plots_per_line, i % plots_per_line
        # remove xlable everywhere except last row
        if not line == nb_lines - 1 and remove_unnecessary_xlabel:
            ax.set_xlabel(None)
        # remove ylabel everywhere except first column
        if not col == 0:
            if remove_unnecessary_ylabel:
                ax.set_ylabel(None)
            if remove_unnecessary_yticks:
                ax.set_yticks([])
        # remove twinx ylabel everywhere except last column
        if twinx and not col == plots_per_line - 1:
            if remove_unnecessary_twinxlabel:
                twinx_axis[i].set_ylabel(None)
            if remove_unnecessary_twinxticks:
                twinx_axis[i].set_yticks([])
        if titles is not None:
            ax.set_title(titles[i])
    if legend_outside and plot_legend[0]:
        ax0 = axs[0, 0] if nb_lines > 1 else axs[0]
        if legend_handles[0] is not None:
            handles_labels = legend_handles[0]
        else:
            handles_labels = ax0.get_legend_handles_labels()
        fig.legend(*handles_labels, **legend_kwargs)
    if twinx and legend_outside and plot_legend[1]:
        ax1 = twinx_axis[0]
        if legend_handles[1] is not None:
            handles_labels = legend_handles[1]
        else:
            handles_labels = ax1.get_legend_handles_labels()
        if "labels" in twinx_legend_kwargs:
            handles_labels = (handles_labels[0], twinx_legend_kwargs["labels"])
            del twinx_legend_kwargs["labels"]
        fig.legend(*handles_labels, **twinx_legend_kwargs)
    if suptitle is not None and plot_suptitle:
        fig.suptitle(suptitle, **suptitle_kwargs)
    if fname_set is not None:
        fname = fname_set
    plt.tight_layout()
    if fname is not None:
        path = os.path.dirname(fname)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(fname, bbox_inches='tight')
    plt.close()




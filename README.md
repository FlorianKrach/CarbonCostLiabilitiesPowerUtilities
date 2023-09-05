# Modelling of a Power Utility Firm

This is the official implementation of the paper [The Financial Impact of Carbon Emissions on Power Utilities Under Climate Scenarios]().


## Installation & Requirements
This code was executed using Python 3.7.
download repo (if wanted, install miniconda and 
make a new environment). then go to main directory and install needed libraries:
````shell
pip install -r requirements.txt
````


## Usage

- print values from paper:
    ```shell
    python run.py --function=print_values_from_paper
    ```

- power plant cost comparison:
  ```shell
  python run.py --function=plot_power_plant_cost_comparison --config=current_sold_capacity_config_1
  ```

- NGFS scenarios:
    ```shell
    python run.py --function=plot_running_default_prob_and_scenarios --config=ngfs_plot_config
    ```

- plot capacities and energy mix in the 3 energy mix scenarios:
  ```shell
  python run.py --function=plot_capacities_and_energy_mix --config=IRP2030_config_market_cal
  python run.py --function=plot_capacities_and_energy_mix --config=IRP2030_green_continue_config
  python run.py --function=plot_capacities_and_energy_mix --config=IRP2030_aggr_green_continue_config
  ```
  
  joint plot of the 3 energy mix scenarios:
  ```shell
  python run.py --function=multiplot --config=joint_rlz_cap_plot_config
  ```

- plot minimal electricity prices in the 3 energy mix scenarios:
  ```shell
  python run.py --function=plot_min_electricity_prices --config=IRP2030_config_market_cal
  python run.py --function=plot_min_electricity_prices --config=IRP2030_green_continue_config
  python run.py --function=plot_min_electricity_prices --config=IRP2030_aggr_green_continue_config
  ```
  
  joint plot of the 3 energy mix scenarios:
  ```shell
  python run.py --function=multiplot --config=joint_min_elect_price_plot_config
  ```
  

- Base case:
  - calibrate the electricity model scaling factors to the market probabilities of default, with init_assets fixed to market value:
    ```shell
    python run.py --function=fit_electricity_price_scaling_factors --config=IRP2030_config_2
    ```
    
  - calibrate the electricity model scaling factors to the market probabilities of default, with smaller init_assets:
    ```shell
    python run.py --function=fit_electricity_price_scaling_factors --config=IRP2030_config_5
    ```

  - default probabilities and bond prices and with electricity model __calibrated to the market probability of default__
    ```shell
    python run.py --function=plot_mean_electricity_prices_for_MPPJ --config=IRP2030_config_market_cal
    python run.py --function=multiplot --config=joint_fitdefaultprobs_meanelecprice_plot_config
    python run.py --function=plot_running_default_prob_and_scenarios --config=IRP2030_config_market_cal
    python run.py --function=plot_bond_price_term_structure --config=IRP2030_config_market_cal
    python run.py --function=plot_running_default_prob_and_scenarios --config=IRP2030_config_market_cal1
    python run.py --function=multiplot --config=joint_defprob_bondprice_plot_config
    ```

- Green continuation: to decrease coal and increase renewables
  ```shell
  python run.py --function=plot_running_default_prob_and_scenarios --config=IRP2030_green_continue_config
  python run.py --function=plot_bond_price_term_structure --config=IRP2030_green_continue_config
  python run.py --function=plot_running_default_prob_and_scenarios --config=IRP2030_green_continue_config1
  ``` 


- Aggressive green continuation: with const coal and increase renewables & gas
  ```shell
  python run.py --function=plot_running_default_prob_and_scenarios --config=IRP2030_aggr_green_continue_config
  python run.py --function=plot_bond_price_term_structure --config=IRP2030_aggr_green_continue_config
  python run.py --function=plot_running_default_prob_and_scenarios --config=IRP2030_aggr_green_continue_config1
  ```
  
  joint plot:
  ```shell
  python run.py --function=multiplot --config=joint_defprob_instdefprob_bondprice_plot_config
  ```





# Harvard CS91r

## Summary

A semester long independent research project for the Spring 2020 term. This repo contains some first steps towards modeling the low-carbon energy transition.

Student: Alaisha Sharma

Advisors: Jackson Killian, Andrew Perrault

Faculty: Milind Tambe

## Notation

- rplant : renewable plant 
- fplant : fossil fuel plant 
- RES (or res) : referring to renewable plant 
- FF (or ff) : referring to fossil fuel plant 

## Running

To run any of the MDP model versions implemented, first create a .txt file containing the parameters for the appropriate version model in Python dictionary format. Any parameters that depend on tech stage should be written as a tuple of length `n_techstages`. 

Run the following command, specifying the model version and passing in the .txt file for parameters. Any file name may be used for the output file, which is where the parameters used and optimal policy produced after running the MDP will be printed.

```
$ python run_mdp_fh -m [model_version] -p [params_file] -o [output_file]
```

To see a breakdown of cost in versions 2 or higher, use a similar .txt file containing the parameters as above. The matrix that will be written to the output file will calculate a particular cost component as an absolute (these numbers will be positive although the cost is negated in the actual rewards matrix) and as a percentage of the total cost.

Run the following command, specifying the options as above. The final option, component, may be one of the following choices:
- rplants_cap
- fplants_OM
- fplants_OM_fix
- fplants_OM_var
- co2_emit
- co2_tax
- storage_cap
- storage_OM
- storage_OM_fix
- storage_OM_var

```
$ python calculate_partial_costs -m [model_version] -p [params_file] -o [output_file] -c [component]
```

In the output file, the state is always written as `(t, v, r)`, where `t` is current time in years (and thus current carbon tax), `v` is current tech stage, and `r` is current number out of total plants that are renewable. 

## Progress

These Markov Decision Problem (MDP) models use the Finite Horizon optimal policy algorithm. Built using Python's MDP Toolbox.

### MDP v0

VARIABLES
- `n_years`: Total time period (number of steps).
- `n_techstages`: Number of technological advancement stages it is possible to pass through.
- `n_plants`: Total number of power plants. All begin as fossil fuel plants and can be converted to renewable.
- `plant_size`: Nameplate capacity of single power plant. 
- `plant_capacity`: Capacity factor of single power plant.
- `c_co2_init`: Starting price of carbon per ton.
- `co2_inc`: Increment of carbon tax as percent per year.
- `c_cap_res`: Initial construction costs of a renewable plant per kW (avg of solar PV and onshore wind). Depends on tech stage.
- `c_om_ff`: Annual operation & maintenance costs of a fossil fuel plant per kW (avg of coal and natural gas). Independent of tech stage.
- `ff_emit`: Annual emissions of a fossil fuel plant in kg CO2 per kWh (avg of coal and natural gas).
- `p_adv_techstage`: Probability that tech stage advances to the next given the current stage is not the highest. Assume it is only possible to advance by 1 at a time.
- `p_rplant_fail` = 1 / `rplant_life`: Probability that a renewable plant "fails" at the end of the year. A plant that fails is always replaced in the next year for the same cost as building a new plant.
- `disc_rate`: Discount rate (avg of solar PV and onshore wind rates in North America).

ASSUMPTIONS
- Carbon tax is entirely deterministic given time (instead of subject to uncertainty in current politics).
- Operation & maintenance costs of fossil fuel power plants does not change as tech stage advanges.
- Renewable power plants have zero operation & maintenance costs.
- Renewable power plant construction incurs zero CO2 emissions.
- Cost of replacing failed renewable power plant is equal to cost of building new plant in current tech stage. 
- Renewable power plant lifetime (and resulting need for replacement) can be modeled as Bernoulli probability of failure each year (probability is reciprocal of lifetime).
- Fossil fuel and renewable power plants have the same capacity factor and are built to the same generation size.

### MDP v1

VARIABLES
- `n_years`: Total time period (number of steps).
- `n_techstages`: Number of technological advancement stages it is possible to pass through.
- `n_plants`: Total number of power plants. All begin as fossil fuel plants and can be converted to renewable.
- `fplant_size`: Nameplate capacity of single fossil fuel plant. 
- `fplant_capacity`: Capacity factor of single fossil fuel plant.
- `rplant_capacity`: Capacity factor of single renewable plant. Nameplate capacity (size) of renewable plant to be built can be calculated from other plant size/capacity parameters.
- `c_co2_init`: Starting price of carbon per ton.
- `co2_inc`: Increment of carbon tax as percent per year.
- `c_ff_fix`: Annual fixed operation & maintenance costs of a fossil fuel plant per kW (avg of coal and natural gas). Independent of tech stage.
- `c_ff_var`: Annual variable operation & maintenance costs of a fossil fuel plant per kWh (avg of coal and natural gas). Independent of tech stage.
- `ff_emit`: Annual emissions of a fossil fuel plant in kg CO2 per kWh (avg of coal and natural gas).
- `c_res_cap`: Initial construction costs of a renewable plant per kW (avg of solar PV and onshore wind). Depends on tech stage.
- `c_bss_cap`: Initial construction costs of a 4h li-ion battery system per kW. Depends on tech stage.
- `c_bss_fix`: Annual fixed operation & maintenance costs of a 4h li-ion battery system plant per kWh. Independent of tech stage.
- `c_bss_var`: Annual variable operation & maintenance costs of a 4 hour li-ion battery system plant per kWh. Independent of tech stage.
- `p_adv_techstage`: Probability that tech stage advances to the next given the current stage is not the highest. Assume it is only possible to advance by 1 at a time.
- `p_rplant_fail`: Probability that a renewable plant "fails" at the end of the year. A plant that fails is always replaced in the next year for the same cost as building a new plant.
- `disc_rate`: Discount rate (avg of solar PV and onshore wind rates in North America).

(ADDITIONAL) ASSUMPTIONS
- Storage capacity required grows exponentially with percent renewable penetration.
- Construction costs of battery system combine energy vs. power dependent costs.

### MDP v2

VARIABLES
- `n_years`: Total time period (number of steps).
- `n_techstages`: Number of technological advancement stages it is possible to pass through.
- `n_plants`: Total number of power plants. All begin as fossil fuel plants and can be converted to renewable.
- `fplant_size`: Nameplate capacity of single fossil fuel plant. 
- `fplant_capacity`: Capacity factor of single fossil fuel plant.
- `rplant_capacity`: Capacity factor of single renewable plant. Nameplate capacity (size) of renewable plant to be built can be calculated from other plant size/capacity parameters.
- `rplant_lifetime`: Number of years that a single renewable power plant can run before needing renewal construction. 
- `c_co2_init`: Starting price of carbon per ton.
- `co2_inc`: Increment of carbon tax as percent per year.
- `c_ff_fix`: Annual fixed operation & maintenance costs of a fossil fuel plant per kW (avg of coal and natural gas). Independent of tech stage.
- `c_ff_var`: Annual variable operation & maintenance costs of a fossil fuel plant per kWh (avg of coal and natural gas). Independent of tech stage.
- `ff_emit`: Annual emissions of a fossil fuel plant in kg CO2 per kWh (avg of coal and natural gas).
- `c_res_cap`: Initial construction costs of a renewable plant per kW (avg of solar PV and onshore wind). Depends on tech stage.
- `bss_coefs`: Coefficients for the exponential function that models storage required (as % of system load) for a given % renewable penetration.
- `c_bss_cap`: Initial construction costs of a 4h li-ion battery system per kW. Depends on tech stage.
- `c_bss_fix`: Annual fixed operation & maintenance costs of a 4h li-ion battery system plant per kWh. Independent of tech stage.
- `c_bss_var`: Annual variable operation & maintenance costs of a 4 hour li-ion battery system plant per kWh. Independent of tech stage.
- `p_adv_techstage`: Probability that tech stage advances to the next given the current stage is not the highest. Assume it is only possible to advance by 1 at a time.
- `disc_rate`: Discount rate (avg of solar PV and onshore wind rates in North America).

(ADDITIONAL) ASSUMPTIONS
- Renewable plant failure is modeled as annual O&M cost equal to capital cost scaled by renewable plant lifetime.

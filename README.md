# Harvard CS91r

## Summary

A semester long independent research project for the Spring 2020 term. This repo contains some first steps towards modeling the low-carbon energy transition.

Student: Alaisha Sharma

Advisors: Jackson Killian, Andrew Perrault

Faculty: Milind Tambe

## Notation

rplant - renewable plant 
fplant - fossil fuel plant
RES - referring to renewable plant
FF - referring to fossil fuel plant

## Progress

### MDP v0

A simple Markov Decision Problem (MDP) model using the Finite Horizon optimal policy algorithm. Built using Python's MDP Toolbox.

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
- Renewable power plant construction incurs zero emissions.
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

ASSUMPTIONS
- Carbon tax is entirely deterministic given time (instead of subject to uncertainty in current politics).
- Operation & maintenance costs of fossil fuel power plants does not change as tech stage advanges.
- Renewable power plants have zero operation & maintenance costs.
- Renewable power plant construction incurs zero emissions.
- Cost of replacing failed renewable power plant is equal to cost of building new plant in current tech stage. 
- Renewable power plant lifetime (and resulting need for replacement) can be modeled as Bernoulli probability of failure each year (probability is reciprocal of lifetime).
- Storage capacity required grows exponentially with percent renewable penetration.
- Construction costs of battery system combine energy vs. power dependent costs.

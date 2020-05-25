# Harvard CS91r - 2020 Spring

## Summary

A semester long independent research project for the Spring 2020 term. This repo contains some first steps towards modeling the low-carbon energy transition.

Student: Alaisha Sharma

Advisors: Jackson Killian, Andrew Perrault

Faculty: Milind Tambe

## Notation

- FH (or fh): referring to Finite Horizon as used in MDP
- rplant : renewable plant 
- fplant : fossil fuel plant 
- RES (or res) : referring to renewable plant 
- FF (or ff) : referring to fossil fuel plant 
- BS (or bss): referring to battery storage system
- PH (or phs): referring to pumped hydro system

## Variables

- `n_years`: Total time period (number of steps).
- `n_tech_stages`: Number of technological advancement stages it is possible to pass through.
- `n_plants`: Total number of power plants. All begin as FF plants and can be converted to RES plants.
- `ff_size`: Nameplate capacity of single FF plant.
- `ff_capacity`: Capacity factor of single FF plant.
- `res_capacity`: Capacity factor of single RES plant. 
- `res_size`: Nameplate capacity of RES plant in kW can be calculated from given plant size/capacity parameters.
- `res_lifetime`: Number of years that a single RES power plant can run before needing renewal/reconstruction. 
- `c_co2_init`: Starting price of carbon per ton.
- `c_co2_inc`: Increment of carbon tax as percent per year.
- `co2_tax_type`: If "LINEAR", then `c_co2_inc` used as coefficient in linear function to calculate carbon tax. If "EXPONENTIAL", then `c_co2_inc` used as coefficient in exponential function to calculate carbon tax.
- `c_ff_cap`: Initial construction costs of a FF plant (avg of coal and natural gas). Independent of tech stage.
- `c_ff_fix`: Annual fixed operation & maintenance costs of a FF plant (avg of coal and natural gas). Independent of tech stage.
- `c_ff_var`: Annual variable operation & maintenance costs of a FF plant (avg of coal and natural gas). Independent of tech stage.
- `ff_emit`: Annual emissions of a FF plant (avg of coal and natural gas).
- `c_res_cap`: Initial construction costs of a RES plant (avg of solar PV and onshore wind). Dependent on tech stage.
- `storage_coefs`: Coefficients for the exponential function that models storage required (as % of system load) for a given % renewable penetration.
- `storage_mix`: Percentage of storage system built using BSS and PHS, respectively.
- `bss_hrs`: Number of hours that BS system can hold charge for.
- `c_bss_cap`: Initial construction costs of a BS system. Dependent on tech stage.
- `c_bss_fix`: Annual fixed operation & maintenance costs of a BS system per kWh. Independent of tech stage.
- `c_bss_var`: Annual variable operation & maintenance costs of a BS system per kWh. Independent of tech stage.
- `c_phs_cap`: Initial construction costs of a PH system. Independent of tech stage.
- `c_phs_fix`: Annual fixed operation & maintenance costs of a PH system. Independent of tech stage.
- `p_adv_tech`: Probability that tech stage advances to the next given the current stage is not the highest. Assume it is only possible to advance by 1 at a time.
- `p_rplant_fail`: Probability that a RES plant "fails" at the end of the year. A plant that fails is always replaced in the next year for the same cost as building a new plant.
- `disc_rate`: Discount rate (avg of solar PV and onshore wind rates in North America).

The following table describes which variables are passed in as parameters to each model version.

| Variable          |   Unit  | v0 | v1 | v2 | v3 |
| ----------------- | ------- | -- | -- | -- | -- |
| n_years           |         | X  | X  | X  | X  |
| n_tech_stages     |         | X  | X  | X  | X  |
| n_plants          |         | X  | X  | X  | X  |
| ff_capacity       |    %    |    | X  | X  | X  |
| ff_size           |   kW    |    | X  | X  | X  |
| res_capacity      |    %    |    | X  | X  | X  |
| res_size          |   kW    |    | X  | X  | X  |
| res_lifetime      |   yr    |    |    | X  | X  |
| c_co2_init        |    $    | X  | X  | X  | X  |
| c_co2_inc         |    %    | X  | X  | X  | X  |
| co2_tax_type      | lin/exp |    |    |    | X  |
| c_ff_cap          |  $/kW   |    |    | X  | X  |
| c_ff_fix          |  $/kW   | X  | X  | X  | X  |
| c_ff_var          |  $/kWh  | X  | X  | X  | X  |
| ff_emit           | ton/kWh | X  | X  | X  | X  |
| c_res_cap         |  $/kW   | X  | X  | X  | X  |
| storage_mix       |   %,%   |    |    | X  | X  |
| storage_coefs     |         |    |    | X  | X  |
| bss_hours         |   hr    |    |    | X  | X  |
| c_bss_cap         |  $/kWh  |    | X  | X  | X  |
| c_bss_fix         |  $/kW   |    | X  | X  | X  |
| c_bss_var         |  $/kWh  |    | X  | X  | X  |
| c_phs_cap         |  $/kWh  |    |    | X  | X  |
| c_phs_fix         |  $/kW   |    |    | X  | X  |
| p_adv_tech        |         | X  | X  | X  | X  |
| p_rplant_fail     |         | X  | X  |    |    |
| disc_rate         |    %    | X  | X  | X  | X  |

NOTE: In MDP v0, there is a single average `plant_size` and `plant_capacity` used instead of the breakdown by plant type, and in MDP v1, `bss_hrs` is fixed at 4. In MDP v2, `c_phs_fix` is assumed to be 0. In all versions except MDP v3, `p_adv_tech` is a constant.

## Models

Markov Decision Problem (MDP) models using the Finite Horizon optimal policy algorithm. Built using Python's MDP Toolbox. 

The underlying problem formulation is to model an energy system that (hopefully) transitions from 100% FF plants to 100% RES plants. Over this time period, the MDP models how many RES plants are constructed and in which years. 

The "tech stage" is meant to model how technology advances unpredictably over time. Variables that depend on the tech stage are cost related, and usually decrease with tech stage. For instance, we would expect the cost of constructing an RES plant to decrease with time as RES technological efficiency improves.

The FF and RES plants modeled are generic representations of fossil fuel and renewable power plants in the real world. FF plants average natural gas and coal, while RES plants average solar PV and onshore wind. 

All incentives are cost related. There is a carbon tax on CO2 emissions that grows by a small percentage each year, as well as the capital and operation & maintenance costs of FF, RES, and BSS components of the system. 

UNIVERSAL ASSUMPTIONS:
- Energy system is closed and has a fixed demand over the given time period.
- It is only possible to "jump" forward by 1 tech stage at a time.
- Carbon tax is entirely deterministic given time (instead of subject to uncertainty in political climate).
- Operation & maintenance costs of FF plants do not change as tech stage advanges.
- RES plants have zero operation & maintenance costs.
- RES construction incurs zero CO2 emissions.
- Cost of replacing failed RES power plant is equal to cost of building new plant in current tech stage.

NOTE: Model version specific assumptions are noted below.

### MDP v0

This is the simplest version of the MDP. The only components are FF and RES plants. These are modeled equivalently except for cost and plant failure, which addresses plant lifetime. FF plants never fail, but RES plants have some probability of failing at any given time step.  

v0 ASSUMPTIONS:
- FF and RES plants have the same capacity factor and are built to the same generation size.
- RES plant lifetime (and resulting need for replacement) can be modeled as Bernoulli probability of failure each year (probability is reciprocal of lifetime).
- If a RES plant fails, it fails at the beginning of year t and is immediately replaced (can resume normal operations by the end of the year).
- Probability of transitioning from tech stage `v` to `v+1` is independent of `v` and time.
- Carbon tax grows exponentially with time.

### MDP v1

This version attempts to capture how RES penetration of the grid leads to an increasingly unstable electricity supply by adding a battery storage component. Some storage must be built alongside a RES plant, where BSS size is determined as a function of how many RES plants have been built. Plant failure is modeled as in v0 except when a RES plant fails its storage must also be rebuilt.

v1 ASSUMPTIONS:
- Storage capacity required grows exponentially with percent RES penetration.
- Construction costs of BSS combine energy vs. power dependent costs.
- BSS does not persist if its accompanying RES plant fails and its cost of rebuilding include capital costs.
- All BSS costs based on 4h li-ion battery system.

### MDP v2

In this version, plant failure is modeled as an additional annual operation & maintenance cost that represents renewing some portion of the plant each year. RES and FF plants both have this replacement cost, but not storage. Storage is now a customizable mix of batteries (BSS) and pumped hydro (PHS). The function for calculating storage size is parametrized in this version.  

v2 ASSUMPTIONS:
- Storage capacity required grows exponentially with percent RES penetration.
- Construction costs of storage combine energy vs. power dependent costs.
- Operation & maintenance costs of storage account for degradation, so no need for additional replacement cost.
- Capital and operation & maintenance costs of pumped hydro do not change with tech stage.
- RES power plant lifetime (and resulting need for replacement) can be modeled as an additional O&M cost equal to capital cost scaled by RES plant lifetime.
- All BSS costs based on 4h li-ion battery system.

### MDP v3

This version shares most of its core assumptions and structure with the previous version. However, the probabiliy of advancing to the next tech stage can be different for different tech stages. The carbon tax may follow a linear or exponential growth pattern.    

v3 ASSUMPTIONS:
- Probability of transitioning from tech stage `v` to `v+1` is independent of time.

## Results and Visuals

The repo structure for results and visuals, with example filenames, is as follows:
- mdp/
  - results/
    - v3/
      - costs/
        - c_v3_baseline.txt
      - params/
        - p_v3_baseline.txt
      - runs/
        - r_v3_baseline.txt
  - visuals/
    - v3
      - plots/
        - g_v3_opt_policy_total_cost_baseline_0.txt
      - policies/
        - a_v3_one_per_last_ten_years.txt

To run the commands below, first create a .txt file for the parameters or policy as needed. Parameters must be written in Python dictionary format (make sure that the dictionary matches the parameter list of the appropriate MDP model version). Any parameters that depend on tech stage should be written as a tuple of length `n_techstages`. Policies must be written in Python list (make sure the length matches `n_years` in the parameters passed in). The output file need not exist before running the command.

For any commandline option that refers to a filename, pass in only the root identifier. For example, to pass in the parameters file "p_v2_baseline.txt" simply use `-p baseline` instead of `-p p_v2_baseline.txt`. To pass in a matching output file, `-o baseline` should be used. Make sure that the files are named and located in this consistent format.

NOTE: If applicable, in the output file the state is always written as `(t, v, r)`, where `t` is current time in years (and thus current carbon tax), `v` is current tech stage, and `r` is current number out of total plants that are renewable. 


### Running MDP

Run the following command, specifying the model version, parameters file, and output file. After running, the output file should contain the optimal policy produced after running the MDP and the rewards matrix calculated for the given parameters.

```
$ python run_mdp_fh.py -m <model_version> -p <params_file> -o <output_file>
```

### Inspecting Cost

Use this command to calculate a breakdown of cost for following the optimal policy. After running, the output file should contain two matrices: 1) where a cost component is calculated as an absolute (these numbers will be positive although the cost is negated in the actual rewards matrix) and 2) where the cost component is calculated as a percentage of the total cost. Both matrices should be the same size as the MDP rewards matrix.

Run the following command, specifying the model version, parameters file and output file as above. The final option, component, may be one of the following choices:
- co2_tax: carbon tax incurred by FF plants
- fplants_total: total cost of FF plants
- ff_replace: failure/renewal cost of FF plants
- ff_om: operation & maintenance cost of FF plants
- res_total: total cost of RES plants
- res_cap: capital cost of RES plants
- res_replace: failure/renewal cost of RES plants
- bss_total: total cost of BSS
- bss_cap: capital cost of BSS
- bss_om: operation & maintenance cost of BSS
- phs_total: total cost of PHS
- phs_cap: capital cost of PHS
- phs_om: operation & maintenance cost of PHS

```
$ python calc_partial_costs.py -m <model_version> -p <params_file> -c <component>
```

NOTE: All cost components calculated as system totals, not per plant. For invalid actions, absolute costs are displayed as infinity and percentages as -1.0.

### Visualizing Cost

For these commands, if the time range is unspecified, the entire time period of `n_years` will be plotted. If the tech stage is unspecified, then `n_tech_stages` adjacent bars will be plotted at each year. If the policy file is unspecified, costs will be calculated for the optimal policy.

Run the following command to see a single cost component calculated for a single policy (the optimal or an arbitrary policy), specifying the model version and parameters file. The options for component are as listed above.

```
$ python plot_cost_component.py -m <model_version> -p <params_file> -c <component> [-a <policy_file>] [-t time_0 time_N] [-v <tech_stage>]
```

Run the following command to see the cost by year of following a given policy (the optimal or an arbitrary policy), specifying the model version and parameters file. The generated plots will show 1) aggregate total cost by year, 2) absolute cost breakdown by year, and 3) percentage cost breakdown by year.

```
$ python plot_single_policy_costs.py -m <model_version> -p <params_file> [-t time_0 time_N] [-v <tech_stage>] [-a <policy_file>] [--granular]
```

To generate the same three plots for two different policies, run the following command, specifying the model version and parameters file as above. Here if the second policy file is unspecified then the first policy file is compared to the optimal policy. The two policies will be plotted as adjacent bars and if the tech stage is unspecified then as `n_tech_stages` pairs of adjacent bars.

```
$ python plot_double_policy_costs.py -m <model_version> -p <params_file> [-t time_0 time_N] [-v <tech_stage>] -a <policy_file_1> [<policy_file_2>]
```

NOTE: Unless the `granular` option is turned on, only the following cost components will be shown:
- co2_tax
- ff_total
- res_total
- bhs_total
- phs_total

### Visualizing Stochasticity

The following commands generate results by averaging the optimal policy for a number of iterations. This is to capture some of the stochasticity built into the model by probabilistic tech stage transitions. Unless the tech stage is specified in the plot, assume that the results shown are for an averaged optimal policy. For these commands, if the time range is unspecified, the entire time period of `n_years` will be plotted, and if the number of iterations is unspecified, the model will be run 200 times.

To see how a series of reductions in storage costs affect the average optimal policy, run the following command, specifying the model version, parameters file, and storage reductions as a series of arguments. Each argument should be a float representing the desired cost fraction (use 1.0 as the first argument to see a cost curve for a 0% reduction in storage). The annual budget and target RES penetration (percentage of plants that are renewable) should be scalars, given in USD and % respectively, that will be added to the plot to help determine which cost curves are within budget and achieve desired RES penetration. The generated plots will show 1) annual costs, 2) cumulative costs, and 3) number of RES plants (new and existing) for each storage reduction over time.

```
$ python plot_storage_sensitivity.py -m <model_version> -p <params_file> -s <storage_cost_reductions> [-t time_0 time_N] [-i <iterations>] [-b <annual_budget>] [-r <target_RES_penetration>]
```

Run the following command, specifying the model version and parameters file as above, to see both deterministic and stochastic plots of the optimal policy. The generated plots will show 1) how many new RES plants are built under a fixed tech stage, 2) how many new and existing RES plants there are under a fixed tech stage, and 3) the average number of RES plants (new and existing) as well as the average tech stage over time when the model is run `iterations` times.

```
$ python plot_techstage_transition.py -m <model_version> -p <params_file> [-t time_0 time_N] [-i <iterations>]
```

Run the following command, specifying the model version and parameters file as above, to see how CO2 emissions and the CO2 tax change over time when the model is run `iterations` times. The generated plots will show the average number of RES plants (new and existing) against 1) annual emissions and carbon tax cost, and 2) cumulative emissions and carbon tax cost over time.

```
$ python plot_co2_impacts.py -m <model_version> -p <params_file> [-t time_0 time_N] [-i <iterations>]
```

To see the annual CO2 emissions as calculated above but for the optimal policy under different sets of parameters, run the following command, specifying the model version as above. Provide at least one argument to the `-p` option in order to see a curve on the plot. In a second plot, this command also visualizes RES penetration for the optimal policy under the different sets of parameters.

```
$ python plot_compare_co2_res.py -m <model_version> -p [<params_file_1> <params_file_2> <params_file_3> ...] [-t time_0 time_N] [-i <iterations>]
```

Finally, run the following command, specifying the model version as above, to compare the total annual and cumulative costs for the optimal policy under different sets of parameters. Provide at least one argument to the `-p` option in order to see a curve on the plot.

```
$ python plot_compare_total_costs.py -m <model_version> -p [<params_file_1> <params_file_2> <params_file_3> ...] [-t time_0 time_N] [-i <iterations>]
```

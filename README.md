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
- BSS (or bss): referring to battery storage system

## Variables

- `n_years`: Total time period (number of steps).
- `n_tech_stages`: Number of technological advancement stages it is possible to pass through.
- `n_plants`: Total number of power plants. All begin as FF plants and can be converted to RES plants.
- `fplant_size`: Nameplate capacity of single FF plant in kW.
- `fplant_capacity`: Capacity factor of single FF plant.
- `rplant_capacity`: Capacity factor of single RES plant. 
- `rplant_size`: Nameplate capacity of RES plant in kW can be calculated from given plant size/capacity parameters.
- `rplant_lifetime`: Number of years that a single RES power plant can run before needing renewal/reconstruction. 
- `c_co2_init`: Starting price of carbon per ton.
- `co2_inc`: Increment of carbon tax as percent per year.
- `c_ff_fix`: Annual fixed operation & maintenance costs of a FF plant per kW (avg of coal and natural gas). Independent of tech stage.
- `c_ff_var`: Annual variable operation & maintenance costs of a FF plant per kWh (avg of coal and natural gas). Independent of tech stage.
- `ff_emit`: Annual emissions of a FF plant in kg CO2 per kWh (avg of coal and natural gas).
- `c_res_cap`: Initial construction costs of a RES plant per kW (avg of solar PV and onshore wind). Depends on tech stage.
- `bss_coefs`: Coefficients for the exponential function that models storage required (as % of system load) for a given % renewable penetration.
- `c_bss_cap`: Initial construction costs of a BSS per kW. Depends on tech stage.
- `c_bss_fix`: Annual fixed operation & maintenance costs of a BSS plant per kWh. Independent of tech stage.
- `c_bss_var`: Annual variable operation & maintenance costs of a BSS plant per kWh. Independent of tech stage.
- `p_adv_techstage`: Probability that tech stage advances to the next given the current stage is not the highest. Assume it is only possible to advance by 1 at a time.
- `p_rplant_fail`: Probability that a RES plant "fails" at the end of the year. A plant that fails is always replaced in the next year for the same cost as building a new plant.
- `disc_rate`: Discount rate (avg of solar PV and onshore wind rates in North America).

The following table describes which variables are passed in as parameters to each model version.

| Variable          | v0 | v1 | v2 |
| ----------------- | -- | -- | -- |
| n_years           | X  | X  | X  |
| n_tech_stages     | X  | X  | X  |
| n_plants          | X  | X  | X  |
| fplant_capacity   |    | X  | X  |
| fplant_size       |    | X  | X  |
| rplant_capacity   |    | X  | X  |
| rplant_size       |    | X  | X  |
| rplant_lifetime   |    |    | X  |
| c_co2_init        | X  | X  | X  |
| co2_inc           | X  | X  | X  |
| c_ff_fix          | X  | X  | X  |
| c_ff_var          | X  | X  | X  |
| ff_emit           | X  | X  | X  |
| c_res_cap         | X  | X  | X  |
| bss_coefs         |    |    | X  |
| c_bss_cap         |    | X  | X  |
| c_bss_fix         |    | X  | X  |
| c_bss_var         |    | X  | X  |
| p_adv_techstage   | X  | X  | X  |
| p_rplant_fail     | X  | X  |    |
| disc_rate         | X  | X  | X  |

NOTE: In MDP v0, there is a single average `plant_size` and `plant_capacity` used instead of the breakdown by plant type, and `c_ff_var` is assumed to be 0.

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

### MDP v1

This version attempts to capture how RES penetration of the grid leads to an increasingly unstable electricity supply by adding a battery storage component. Some storage must be built alongside a RES plant, where BSS size is determined as a function of how many RES plants have been built. Plant failure is modeled as in v0 except when a RES plant fails its storage must also be rebuilt.

v1 ASSUMPTIONS:
- Storage capacity required grows exponentially with percent RES penetration.
- Construction costs of BSS combine energy vs. power dependent costs.
- BSS does not persist if its accompanying RES plant fails and its cost of rebuilding include capital costs.
- All BSS costs based on 4h li-ion battery system.

### MDP v2

In this version, plant failure is modeled differently because failing BSS alongside RES plants is unrealistic. Instead of needing to rebuilt, RES plants now have an operation & maintenance cost that represents renewing some portion of the plant each year. The function for calculating BSS size is parametrized in this version.  

v2 ASSUMPTIONS:
- Storage capacity required grows exponentially with percent RES penetration.
- Construction costs of BSS combine energy vs. power dependent costs.
- RES power plant lifetime (and resulting need for replacement) can be modeled as an additional O&M cost equal to capital cost scaled by RES plant lifetime.
- All BSS costs based on 4h li-ion battery system.

## Results and Visuals

The repo structure for results and visuals, with example filenames, is as follows:
- mdp/
  - results/
    - v2/
      - costs/
        - c_v2_baseline.txt
      - params/
        - p_v2_baseline.txt
      - runs/
        - r_v2_baseline.txt
  - visuals/
    - v2
      - plots/
        - g_v2_opt_policy_total_cost_baseline_0.txt
      - policies/
        - a_v2_one_per_third_year.txt

To run the commands below, first create a .txt file for the parameters or policy as needed. Parameters must be written in Python dictionary format (make sure that the dictionary matches the parameter list of the appropriate MDP model version). Any parameters that depend on tech stage should be written as a tuple of length `n_techstages`. Policies must be written in Python list (make sure the length matches `n_years` in the parameters passed in). The output file need not exist before running the command.

For any commandline option that refers to a filename, pass in the root identifier. For example, to pass in the parameters file "p_v2_baseline.txt" simply use `-p baseline` instead of `-p p_v2_baseline.txt`. To pass in a matching output file, `-o baseline` should be used. Make sure that the files are named and located in this consistent format.

NOTE: If applicable, in the output file the state is always written as `(t, v, r)`, where `t` is current time in years (and thus current carbon tax), `v` is current tech stage, and `r` is current number out of total plants that are renewable. 


### Running MDP

Run the following command, specifying the model version, parameters file, and output file. After running, the output file should contain the optimal policy produced after running the MDP and the rewards matrix calculated for the given parameters.

```
$ python run_mdp_fh.py -m <model_version> -p <params_file> -o <output_file>
```

### Inspecting Cost

Use this command to calculate a breakdown of cost in any of the MDP model versions 2+. After running, the output file should contain two matrices: 1) where a cost component is calculated as an absolute (these numbers will be positive although the cost is negated in the actual rewards matrix) and 2) where the cost component is calculated as a percentage of the total cost. Both matrices should be the same size as the MDP rewards matrix.

Run the following command, specifying the model version, parameters file and output file as above.The final option, component, may be one of the following choices:
- rplants_total: total cost of RES plants
- rplants_cap: capital cost of RES plants
- rplants_replace: failure/renewal cost of RES plants
- fplants_total: total cost of FF plants
- fplants_OM: operation & maintenance cost of FF plants
- fplants_OM_fix: fixed operation & maintenance cost of FF plants
- fplants_OM_var: variable operation & maintenance cost of FF plants
- co2_emit: CO2 emissions from FF plants
- co2_tax: carbon tax incurred by FF plants
- storage_total: total cost of BSS
- storage_cap: capital cost of BSS
- storage_OM: operation & maintenance cost of BSS
- storage_OM_fix: fixed operation & maintenance cost of BSS
- storage_OM_var: variable operation & maintenance cost of BSS

```
$ python calc_partial_costs.py -m <model_version> -p <params_file> -o <output_file> -c <component>
```

NOTE: All cost components calculated as system totals, not per plant. For invalid actions, absolute costs are displayed as infinity and percentages as -1.0.

### Visualizing Cost

Use this command to see the cost by year of following the optimal policy. 

Run the following command, specifying the model version, tech stage (which remains fixed to reduce the number of dimensions), and parameters file. The generated plots will show 1) aggregate total cost by year, 2) absolute cost breakdown by year, and 3) percentage cost breakdown by year.

```
$ python plot_opt_policy_costs.py -m <model_version> -p <params_file> -v <tech_stage>
```

To generate the same three plots for an arbitrary policy, run the following command, specifying the model version, tech stage, and parameters file as above, as well as the policy file. Note that this command will calculate costs for the entire policy relative to a single tech stage, unlike the optimal policy above.   

```
$ python plot_other_policy_costs.py -m <model_version> -p <params_file> -a <policyfile> -v <tech_stage>
```

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pylab as plt
import seaborn as sns
sns.set()
plt.style.use("seaborn-darkgrid")
import datetime
import os
import arviz as az
az.style.use("arviz-whitegrid")
az.rcParams["stats.hdi_prob"] = 0.95
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.tsa.stattools as smt
import pymc3 as pm

df = pd.read_csv("/Users/Documents/TMDU/COVID19/covid19_domestic_daily_data.csv")

df = df.iloc[0:685]
df.loc[684]
#to time-series data
df["date"]  = pd.to_datetime(df["date"])
#duration
start = pd.Period("2020-01-16", freq = "D")
end = pd.Period("2021-11-30", freq = "D")

#total
Total = df.copy()
Total = Total.reset_index()
Total = Total.drop("index", axis = 1)
ts = Total.set_index("date").groupby(level = 0)

#NPIs
GoTo_start = datetime.date(2020, 7, 22)
GoToEat_start = datetime.date(2020, 10, 1)
SE1_start = datetime.date(2020,4,7)
SE1_end = datetime.date(2020,5,25)
SE2_start = datetime.date(2021,1,7)
SE2_end = datetime.date(2021,3,21)
SE3_start = datetime.date(2021,4,25)
SE3_end = datetime.date(2021,6,20)
SE4_start = datetime.date(2021,7,12)
SE4_end = datetime.date(2021,9,30)
school_closure_start = datetime.date(2020, 3, 2)
school_closure_end = datetime.date(2020, 4, 5)
Goto_end = datetime.date(2020, 12, 28)
anl_start = datetime.date(2020, 1, 16)
anl_end = datetime.date(2021, 11, 30)

GoTo_start - anl_start#188
GoToEat_start - anl_start#259
Goto_end - anl_start#347

GoTo_start_day = 188
GoToEat_start_day = 259
GoTo_end_day = 347
GoTo_duration = GoTo_end_day - GoTo_start_day
GoTo_duration_tokyo = GoTo_end_day - GoToEat_start_day

SE1_start-anl_start#82
SE1_end-anl_start#130
SE2_start-anl_start#357
SE2_end-anl_start#430
SE3_start-anl_start#465
SE3_end-anl_start#521
SE4_start-anl_start#543
SE4_end-anl_start#623

school_closure_start-anl_start#46
school_closure_end-anl_start#80

SE1_start_day = 82
SE1_end_day = 130
SE2_start_day = 357
SE2_end_day = 430
SE3_start_day = 465
SE3_end_day = 521
SE4_start_day = 543
SE4_end_day = 623

school_closure_start_day = 46
school_closure_end_day = 80

date = pd.date_range("2020-01-16", "2021-11-30")
N = len(date)

#death
with pm.Model() as infection_model:
    #Prior Distributions
    mu_delay = pm.Normal("mu_delay", mu = 21.82, sigma = 1.01)#mean: infection to death
    dispersion_delay = pm.Normal("dispersion_delay", mu = 14.26, sigma = 5.18)#dispersion: infection to death
    delay = pm.NegativeBinomial("delay", mu = mu_delay, alpha = dispersion_delay, shape = 2)#delay time to effect each NPI
    #NPI parameters
    para_sigma = pm.HalfStudentT("para_sigma", nu = 3, sigma = 0.04, shape = 2)
    para_mean_npi = pm.AsymmetricLaplace("para_mean_npi", b = 10, kappa = 0.5, mu = 0)
    para_mean_es = pm.AsymmetricLaplace("para_mean_es", b = 10, kappa = 2, mu = 0)
    goto = pm.Normal("goto", mu = para_mean_es, sigma = para_sigma[0], shape = 2)
    se = pm.Normal("se", mu = para_mean_npi, sigma = para_sigma[1], shape = 4)
    sc = pm.Normal("sc", mu = para_mean_npi, sigma = para_sigma[1])
    #NPIs
    npi1 = pm.math.switch((GoTo_start_day+delay[0] <= np.arange(0, N)) & (np.arange(0, N) <= GoTo_end_day+delay[0]), goto[0], 0)
    npi2 = pm.math.switch((GoToEat_start_day+delay[0] <= np.arange(0, N)) & (np.arange(0, N) <= GoTo_end_day+delay[0]), goto[1], 0)
    npi3 = pm.math.switch((SE1_start_day+delay[1] <= np.arange(0, N)) & (np.arange(0, N) <= SE1_end_day + delay[1]), se[0], 0)
    npi4 = pm.math.switch((SE2_start_day+delay[1] <= np.arange(0, N)) & (np.arange(0, N) <= SE2_end_day + delay[1]), se[1], 0)
    npi5 = pm.math.switch((SE3_start_day+delay[1] <= np.arange(0, N)) & (np.arange(0, N) <= SE3_end_day + delay[1]), se[2], 0)
    npi6 = pm.math.switch((SE4_start_day+delay[1] <= np.arange(0, N)) & (np.arange(0, N) <= SE4_end_day + delay[1]), se[3], 0)
    npi7 = pm.math.switch((school_closure_start_day+delay[1] <= np.arange(0, N)) & (np.arange(0, N) <= school_closure_end_day + delay[1]), sc, 0)
    npi = pm.Deterministic("npi", npi1+npi2+npi3+npi4+npi5+npi6+npi7)
    cnpi = pm.Deterministic("cnpi", npi1+npi3+npi4+npi5+npi6+npi7)
    #actual rate
    r = pm.Exponential("r", 1.0/ts["death"].sum().mean(), shape = N)
    r_e = pm.Deterministic("r_e", r*pm.math.exp(-npi))
    r_c = pm.Deterministic("r_c", r*pm.math.exp(-cnpi))
    infection = pm.Poisson("obs", r_e, observed = ts["death"].sum())
    trace = pm.sample(10000, tune = 5000, cores = 4, nuts = {"target_accept":0.99}, return_inferencedata = False)

varnames = ["goto", "se", "sc"]
pm.summary(trace, varnames)
np.median(trace["goto"][:, 0])
az.plot_trace(trace, varnames, compact = False)
az.plot_forest(trace, r_hat = True)
pm.model_to_graphviz(infection_model).render("/Users/tk/Desktop/model")

#violin_plot
df_violin_1 = pd.DataFrame({"Posterior median reduction in r$_t$":-trace["goto"][:,0], "NPI":"Go To"})
df_violin_2 = pd.DataFrame({"Posterior median reduction in r$_t$":-trace["goto"][:,1], "NPI":"Go To(Tokyo)"})
df_violin_3 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["se"][:, 0], "NPI":"State of Emergency 1"})
df_violin_4 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["se"][:, 1], "NPI":"State of Emergency 2"})
df_violin_5 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["se"][:, 2], "NPI":"State of Emergency 3"})
df_violin_6 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["sc"], "NPI":"School Closure"})
df_violin =  pd.concat([df_violin_1, df_violin_2, df_violin_3, df_violin_4, df_violin_5, df_violin_6])
sns.violinplot(x = "Posterior median reduction in r$_t$", y = "NPI" , data = df_violin, inner = "box")
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.title(r"$\leftarrow$""increase infection         reduce infection"r"$\rightarrow$")
plt.savefig("/Users/tk/Desktop/violin_death.png", dpi = 300)

#violin_plot_ES
sns.set_palette("bwr_r")
df_violin_1 = pd.DataFrame({"Posterior median increase in r$_t$":-trace["goto"][:,0], "Economic Stimulus":"After Go To"})
df_violin_2 = pd.DataFrame({"Posterior median increase in r$_t$":-trace["goto"][:,1], "Economic Stimulus":"After Go To Eat"})
df_violin =  pd.concat([df_violin_1, df_violin_2])
sns.violinplot(x = "Posterior median increase in r$_t$", y = "Economic Stimulus" , data = df_violin, inner = "box", aspect = 1.618)
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.xticks([np.log(0.25), np.log(0.5), 0, np.log(2), np.log(4), np.log(8.17)], ["-140%", "-70%", "0%", "70%", "140%", "210%"])
plt.title(r"$\leftarrow$""did not increase COVID-19 death \n increased COVID-19 death"r"$\rightarrow$", fontsize = 16)
plt.savefig("/Users/tk/Desktop/violin_death_es.png", dpi = 300)

#violin_plot_NPIs
sns.set_palette("cool")
df_violin_3 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["se"][:, 0], "NPI":"After SE1"})
df_violin_4 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["se"][:, 1], "NPI":"After SE2"})
df_violin_5 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["se"][:, 2], "NPI":"After SE3"})
df_violin_6 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["se"][:, 3], "NPI":"After SE4"})
df_violin_7 = pd.DataFrame({"Posterior median reduction in r$_t$":trace["sc"], "NPI":"After SC"})
df_violin =  pd.concat([df_violin_3, df_violin_4, df_violin_5, df_violin_6, df_violin_7])
sns.violinplot(x = "Posterior median reduction in r$_t$", y = "NPI" , data = df_violin, inner = "box", aspect = 1.618)
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.xticks([np.log(0.25), np.log(0.5),  0, np.log(2), np.log(4), np.log(8.17)], ["-140%", "-70%", "0", "70%", "140%", "210%"])
plt.title(r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$")
plt.savefig("/Users/tk/Desktop/violin_death_npis.png", dpi = 300)

#plot actual deaths and Byesian estimation graphs togeher.
import math
GoTo_end_day = 347
N1 = math.floor(GoTo_end_day + trace["delay"][:, 1].mean())
sns.relplot(x = date[:N1], y = ts["death"].sum()[:N1], kind = "line", height = 10, aspect = 1.618, label = "actual mortality", color = "r")
plt.subplots_adjust(top = 0.95, bottom = 0.05, left = 0.05)
plt.title("Total", fontsize = 20)
plt.ylabel("Number of death", fontsize = 16)
plt.xlabel("date", fontsize = 16)
plt.fill_between(x = [GoToEat_start, Goto_end], y1 = 0, y2 = max(ts["death"].sum()[:N1]), color = "m", alpha = 0.1, label = "Go To Eat")
counterfactual_death = [trace["r_c"][:, i].mean() for i in range(N)]
plt.plot(date[:N1], counterfactual_death[:N1], "k--", lw = 1, label = "counterfactual mortality")
plt.legend()
plt.savefig("/Users/tk/Desktop/plot_without_Gotoeat.png", dpi = 300)

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
import pymc as pm
import pickle
print(f"Running on PyMC v{pm.__version__}")

df = pd.read_csv("/Users/tk/Documents/TMDU/COVID19/BaseData/NumberOfDeathByPrefecture.csv", header = 0, index_col = 0)
df.mean().sort_values()
not_incolumns = df.columns[~df.columns.isin(["Hokkaido", "Tokyo", "Kanagawa", "Saitama", "Chiba", "Aichi", "Osaka", "Hyogo", "Fukuoka", "Okinawa"])]
df.drop(not_incolumns, axis = "columns", inplace = True)

df = df.iloc[0:685, :]

#to time-series data
df.index = pd.to_datetime(df.index)

#duration
start = pd.Period("2020-01-16", freq = "D")
end = pd.Period("2021-11-30", freq = "D")

#NPIs
GoTo_start = datetime.date(2020, 7, 22)
GoToEat_start = datetime.date(2020, 10, 1)
SE1_start = datetime.date(2020,4,7)
SE1_end = datetime.date(2020,5,25)
SE2_start = datetime.date(2021,1,8)
SE2_end = datetime.date(2021,3,21)
SE3_start = datetime.date(2021,4,25)
SE3_end = datetime.date(2021,9,30)
school_closure_start = datetime.date(2020, 3, 2)
school_closure_end = datetime.date(2020, 4, 5)
GoTo_end = datetime.date(2020, 12, 28)
GoToEat_end = datetime.date(2020, 11, 24)
anl_start = datetime.date(2020, 1, 16)
anl_end = datetime.date(2021, 11, 30)

GoTo_start - anl_start#188
GoToEat_start - anl_start#259
GoTo_end - anl_start#347
GoToEat_end - anl_start#313

GoTo_start_day = 188
GoToEat_start_day = 259
GoTo_end_day = 347
GoToEat_end_day = 313

SE1_start-anl_start#82
SE1_end-anl_start#130
SE2_start-anl_start#358
SE2_end-anl_start#430
SE3_start-anl_start#465
SE3_end-anl_start#623

school_closure_start-anl_start#46
school_closure_end-anl_start#80

SE1_start_day = 82
SE1_end_day = 130
SE2_start_day = 358
SE2_end_day = 430
SE3_start_day = 465
SE3_end_day = 623

school_closure_start_day = 46
school_closure_end_day = 80

date = pd.date_range("2020-01-16", "2021-11-30")
N = len(date)

prefecture_idx, prefectures = pd.factorize(df.columns, sort = False)
coords = {"Prefecture":prefectures, "Date":date}
p = len(prefecture_idx)

#Prefecture adjust
    #SE1
        #start
df_adjust_se1s = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            #Tokyo, Kanagawa, Saitama, Chiba, Osaka, Hyogo, Fukuoka
df_adjust_se1s.loc[:, ["Tokyo", "Kanagawa", "Saitama", "Chiba", "Osaka", "Hyogo", "Fukuoka"]] = SE1_start_day
            #Other area
not_incolumns_se1s = df_adjust_se1s.columns[~df_adjust_se1s.columns.isin(["Tokyo", "Kanagawa", "Saitama", "Chiba", "Osaka", "Hyogo", "Fukuoka"])]
df_adjust_se1s.loc[:, not_incolumns_se1s] = 91#91 = datetime.date(2020,4,16)-anl_start

        #end
df_adjust_se1e = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            #Hokkaido, Tokyo, Kanagawa, Saitama, Chiba
df_adjust_se1e.loc[:, ["Hokkaido", "Tokyo", "Kanagawa", "Saitama", "Chiba"]] = SE1_end_day
            #Osaka, Hyogo
df_adjust_se1e.loc[:, ["Osaka", "Hyogo"]] = 126#126 = datetime.date(2020,5,21)-anl_start
            #Other area
not_incolumns_se1e = df_adjust_se1e.columns[~df_adjust_se1e.columns.isin(["Hokkaido", "Tokyo", "Kanagawa", "Saitama", "Chiba", "Osaka", "Hyogo"])]
df_adjust_se1e.loc[:, not_incolumns_se1e] = 119#119 = datetime.date(2020,5,14)-anl_start

    #SE2
        #start
df_adjust_se2s = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            #Tokyo, Saitama, Chiba, Kanagawa
df_adjust_se2s.loc[:, ["Tokyo", "Saitama", "Chiba", "Kanagawa"]] = SE2_start_day
            #Aichi, Osaka, Hyogo, Fukuoka
df_adjust_se2s.loc[:, ["Aichi", "Osaka", "Hyogo", "Fukuoka"]] = 364#364 = datetime.date(2021, 1,14)-anl_start
not_incolumns_se2s = df_adjust_se2s.columns[~df_adjust_se2s.columns.isin(["Tokyo", "Saitama", "Chiba", "Kanagawa", "Aichi", "Osaka", "Hyogo", "Fukuoka"])]
df_adjust_se2s.loc[:, not_incolumns_se2s] = 2*N
        #end
df_adjust_se2e = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
datetime.date(2021, 2,7)-anl_start
            #Tokyo, Kanagawa, Saitama, Chiba
df_adjust_se2e.loc[:, ["Tokyo", "Kanagawa", "Saitama", "Chiba"]] = SE2_end_day
            #Aichi, Osaka, Hyogo, Fukuoka
df_adjust_se2e.loc[:, ["Aichi", "Osaka", "Hyogo", "Fukuoka"]] = 409#409 = datetime.date(2021,2,28)-anl_start
not_incolumns_se2e = df_adjust_se2e.columns[~df_adjust_se2e.columns.isin(["Tokyo", "Saitama", "Chiba", "Kanagawa", "Aichi", "Osaka", "Hyogo", "Fukuoka"])]
            #Other area
df_adjust_se2e.loc[:, not_incolumns_se2e] = 0

    #SE3 wave 1
        #start
df_adjust_se3s1 = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            #Hokkaido
df_adjust_se3s1.loc[:, ["Hokkaido"]] = 486#486: datetime.date(2021, 5, 16)-anl_start
datetime.date(2021, 5, 16)-anl_start
            #Tokyo, Osaka, Hyogo
df_adjust_se3s1.loc[:, ["Tokyo", "Osaka", "Hyogo"]] = 465#465: datetime.date(2021, 4, 25)-anl_start
datetime.date(2021, 4, 25)-anl_start
            #Aichi, Fukuoka
df_adjust_se3s1.loc[:, ["Aichi", "Fukuoka"]] = 482#482 = datetime.date(2021, 5, 12)-anl_start
datetime.date(2021, 5, 12)-anl_start
            #Okinawa
df_adjust_se3s1.loc[:, ["Okinawa"]] = 493#datetime.date(2021, 5, 23)-anl_start
datetime.date(2021, 5, 23)-anl_start
            #Other area
not_incolumns_se3s1 = df_adjust_se3s1.columns[~df_adjust_se3s1.columns.isin(["Hokkaido", "Tokyo", "Aichi", "Kyoto", "Osaka", "Hyogo", "Okayama", "Hiroshima", "Fukuoka", "Okinawa"])]
df_adjust_se3s1.loc[:, not_incolumns_se3s1] = 2*N
        #end
df_adjust_se3e1 = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            #Hokkaido, Tokyo, Aichi, Osaka, Hyogo, Fukuoka
df_adjust_se3e1.loc[:, ["Hokkaido", "Tokyo", "Aichi", "Osaka", "Hyogo", "Fukuoka"]] = 521#521: datetime.date(2021, 6, 20)-anl_start
datetime.date(2021, 6, 20)-anl_start
            #Okinawa
df_adjust_se3e1.loc[:, ["Okinawa"]] = SE3_end_day
            #Other area
not_incolumns_se3e1 = df_adjust_se3e1.columns[~df_adjust_se3e1.columns.isin(["Hokkaido", "Tokyo", "Aichi", "Osaka", "Hyogo", "Fukuoka", "Okinawa"])]
df_adjust_se3e1.loc[:, not_incolumns_se3e1] = 0

    #SE3 wave 2
        #start
df_adjust_se3s2 = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            #Hokkaido, Aichi
df_adjust_se3s2.loc[:, ["Hokkaido", "Aichi"]] = 589#598 = datetime.date(2021, 8, 27)-anl_start
datetime.date(2021, 8, 27)-anl_start
            #Tokyo
df_adjust_se3s2.loc[:, "Tokyo"] = 543#datetime.date(2021, 7, 12)-anl_start
datetime.date(2021, 7, 12)-anl_start
            #Saitama, Chiba, Kanagawa, Osaka
df_adjust_se3s2.loc[:, ["Chiba", "Saitama", "Kanagawa", "Osaka"]] = 564#datetime.date(2021, 8, 2)-anl_start
datetime.date(2021, 8, 2)-anl_start
            #Hyogo, Fukuoka
df_adjust_se3s2.loc[:, ["Hyogo", "Fukuoka"]] = 582#582 = datetime.date(2021, 8, 20)-anl_start
datetime.date(2021, 8, 20)-anl_start
            #Other area
not_incolumns_se3s2 = df_adjust_se3s2.columns[~df_adjust_se3s2.columns.isin(["Hokkaido", "Tokyo", "Chiba", "Saitama", "Kanagawa", "Osaka", "Hyogo", "Aichi", "Fukuoka"])]
df_adjust_se3s2.loc[:, not_incolumns_se3s2] = 2*N
        #end
df_adjust_se3e2 = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
            #Hokkaido, Saitama, Chiba, Tokyo, Kanagawa, Aichi, Osaka, Hyogo, Fukuoka, Okinawa
df_adjust_se3e2.loc[:, ["Hokkaido", "Saitama", "Chiba", "Tokyo", "Kanagawa", "Aichi", "Osaka", "Hyogo", "Fukuoka", "Okinawa"]] = 623#623: datetime.date(2021, 9, 30)-anl_start
datetime.date(2021, 9, 30)-anl_start
            #Other area
not_incolumns_se3e2 = df_adjust_se3e2.columns[~df_adjust_se3e2.columns.isin(["Hokkaido", "Saitama", "Chiba", "Tokyo", "Kanagawa", "Aichi", "Osaka", "Hyogo", "Fukuoka", "Okinawa"])]
df_adjust_se3e2.loc[:, not_incolumns_se3e2] = 0

    #School Closure
        #start
df_adjust_scs = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_scs.loc[:, :] = 46#46 = datetime.date(2020, 3, 2)-anl_start
datetime.date(2020, 3, 2)-anl_start
        #end
df_adjust_sce = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_sce.loc[:, :] = 80#80 = datetime.date(2020, 4, 5)-anl_start
datetime.date(2020, 4, 5)-anl_start

    #GoToTravel
        #start
df_adjust_travels = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
not_incolumns_travel= df_adjust_travels.columns[~df_adjust_travels.columns.isin(["Tokyo"])]
df_adjust_travels.loc[:, not_incolumns_travel] = GoTo_start_day
df_adjust_travels.loc[:, "Tokyo"] = GoToEat_start_day
        #end
df_adjust_travele = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_travele.loc[:, :] = GoTo_end_day
    #GoToEat
        #start
df_adjust_eats = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_eats.loc[:, :] = GoToEat_start_day
df_adjust_eate = pd.DataFrame(np.zeros([N, p], np.int32), index = df.index, columns = df.columns)
df_adjust_eate.loc[:, :] = GoToEat_end_day

b = np.tile(np.arange(0, N), (p, 1)).T
#death
with pm.Model(coords = coords) as hierarchical_death_model:
    #Prior Distributions
    mu_delay = pm.Normal("mu_delay", mu = 21.82, sigma = 1.01, initval = 21.82)#mean: infection to death
    dispersion_delay = pm.Normal("dispersion_delay", mu = 14.26, sigma = 5.18, initval = 14.26)#dispersion: infection to death
    delay_es = pm.NegativeBinomial("delay_es", mu = mu_delay, alpha = dispersion_delay)
    delay_npi = pm.NegativeBinomial("delay_npi", mu = mu_delay, alpha = dispersion_delay)
    #NPI hyperparameters
    para_sigma_es = pm.HalfStudentT("para_sigma_es", nu = 3, sigma = 0.04)
    para_sigma_npi = pm.HalfStudentT("para_sigma_npi", nu = 3, sigma = 0.04)
    para_mean_es = pm.AsymmetricLaplace("para_mean_es", b = 10, kappa = 2.0, mu = 0)
    para_mean_npi = pm.AsymmetricLaplace("para_mean_npi", b = 10, kappa = 0.5, mu = 0)
    #ES and NPI parameters
    es1 = pm.Normal("es1", mu = para_mean_es, sigma = para_sigma_es, dims = "Prefecture")
    es2 = pm.Normal("es2", mu = para_mean_es, sigma = para_sigma_es, dims = "Prefecture")
    se1 = pm.Normal("se1", mu = para_mean_npi, sigma = para_sigma_npi, dims = "Prefecture")
    se2 = pm.Normal("se2", mu = para_mean_npi, sigma = para_sigma_npi, dims = "Prefecture")
    se3 = pm.Normal("se3", mu = para_mean_npi, sigma = para_sigma_npi, dims = "Prefecture")
    sc = pm.Normal("sc", mu = para_mean_npi, sigma = para_sigma_npi, dims = "Prefecture")
    #NPIs
        #GoToTravel
    scondition_goto_travel = (delay_es < b-df_adjust_travels)
    econdition_goto_travel = (delay_es >= b-df_adjust_travele)
    npi1 = pm.math.switch(scondition_goto_travel & econdition_goto_travel, es1, 0)
        #GoToEat
    scondition_goto_eat = (delay_es < b-df_adjust_eats)
    econdition_goto_eat = (delay_es >= b-df_adjust_eate)
    npi2 = pm.math.switch(scondition_goto_eat & econdition_goto_eat, es2, 0)
        #SE1
    scondition_se1 = (delay_npi < b-df_adjust_se1s)
    econdition_se1 = (delay_npi >= b-df_adjust_se1e)
    npi3 = pm.math.switch(scondition_se1 & econdition_se1, se1, 0)
        #SE2
    scondition_se2 = (delay_npi < b-df_adjust_se2s)
    econdition_se2 = (delay_npi >= b-df_adjust_se2e)
    npi4 = pm.math.switch(scondition_se2 & econdition_se2, se2, 0)
        #SE3 wave 1
    scondition_se31 = (delay_npi < b-df_adjust_se3s1)
    econdition_se31 = (delay_npi >= b-df_adjust_se3e1)
    npi5 = pm.math.switch(scondition_se31 & econdition_se31, se3, 0)
        #SE3 wave 2
    scondition_se32 = (delay_npi < b-df_adjust_se3s2)
    econdition_se32 = (delay_npi >= b-df_adjust_se3e2)
    npi6 = pm.math.switch(scondition_se32 & econdition_se32, se3, 0)
        #SC
    scondition_sc = (delay_npi < b-df_adjust_scs)
    econdition_sc = (delay_npi >= b-df_adjust_sce)
    npi7 = pm.math.switch(scondition_sc & econdition_sc, sc, 0)
        #Combine all npis and es
    npi = pm.Deterministic("npi", npi1+npi2+npi3+npi4+npi5+npi6+npi7)
    cnpi = pm.Deterministic("cnpi", npi3+npi4+npi5+npi6+npi7)
    #actual rate
    r = pm.Exponential("r", 1.0/df.mean(axis = 0), initval = 1.0/df.mean(axis = 0))
    r_e = pm.Deterministic("r_e", r*pm.math.exp(-npi))
    r_c = pm.Deterministic("r_c", r*pm.math.exp(-cnpi))
    #prefectures
    death = pm.Poisson("death", r_e[:, prefecture_idx], observed = df.iloc[:, prefecture_idx])

with hierarchical_death_model:
    idata2 = pm.sample(1000,
                    tune = 500,
                    chains = 4,
                    cores = 8,
                    init = "advi",
                    n_init = 1000,
                    target_accept = 0.99,
                    return_inferencedata = True
                    )

az.summary(idata, var_names = ["delay_es", "delay_npi"])
az.plot_trace(idata, var_names = ["se1"])
pm.save_trace(idata, "model_20230204")
az.summary(idata, var_names = ["se1"], round_to = 2)
az.summary(idata, var_names = ["se2"], round_to = 2)
az.summary(idata, var_names = ["se3"], round_to = 2)
az.summary(idata, var_names = ["sc"], round_to = 2)
az.summary(idata, var_names = ["es1"], round_to = 2)
az.summary(idata, var_names = ["es2"], round_to = 2)
az.summary(idata, var_names = ["r_e"], round_to = 2)
az.summary(idata, var_names = ["r_c"], round_to = 2)
az.summary(idata, var_names = ["delay_es", "delay_npi"], round_to = 2)

#save the trace to disk
with open('model_20230204.pkl', 'wb') as buff:
    pickle.dump(idata, buff)
# Load the trace from disk
with open('model_20230204.pkl', 'rb') as buff:
    idata = pickle.load(buff)

#Summary statistics
np.mean(az.summary(idata, var_names = ["es1"])["hdi_97.5%"])
az.summary(idata, var_names = ["es1"], filter_vars = "like")
az.summary(idata, var_names = ["es1"])["sd"].mean()
az.summary(idata, var_names = ["es1"])["hdi_2.5%"].mean()
az.summary(idata, var_names = ["es1"])["hdi_97.5%"].mean()

az.summary(idata, var_names = ["es2"])["mean"].mean()
az.summary(idata, var_names = ["es2"])["sd"].mean()
az.summary(idata, var_names = ["es2"])["hdi_2.5%"].mean()
az.summary(idata, var_names = ["es2"])["hdi_97.5%"].mean()

az.summary(idata, var_names = ["se1"])["mean"].mean()
az.summary(idata, var_names = ["se1"])["sd"].mean()
az.summary(idata, var_names = ["se1"])["hdi_2.5%"].mean()
az.summary(idata, var_names = ["se1"])["hdi_97.5%"].mean()

az.summary(idata, var_names = ["se2"])["mean"].mean()
az.summary(idata, var_names = ["se2"])["sd"].mean()
az.summary(idata, var_names = ["se2"])["hdi_2.5%"].mean()
az.summary(idata, var_names = ["se2"])["hdi_97.5%"].mean()

az.summary(idata, var_names = ["se3"])["mean"].mean()
az.summary(idata, var_names = ["se3"])["sd"].mean()
az.summary(idata, var_names = ["se3"])["hdi_2.5%"].mean()
az.summary(idata, var_names = ["se3"])["hdi_97.5%"].mean()

az.summary(idata, var_names = ["sc"])["mean"].mean()
az.summary(idata, var_names = ["sc"])["sd"].mean()
az.summary(idata, var_names = ["sc"])["hdi_2.5%"].mean()
az.summary(idata, var_names = ["sc"])["hdi_97.5%"].mean()

az.summary(idata, var_names = ["delay_es"])
az.summary(idata, var_names = ["delay_npi"])
az.plot_forest(idata, var_names = ["es2"], r_hat = True)

pm.model_to_graphviz(hierarchical_death_model).render("/Users/tk/Desktop/model2")

az.plot_trace(idata, var_names = ["es1"], compact = False)
az.plot_trace(idata, var_names = ["es2"], compact = False)
az.plot_trace(idata, var_names = ["se1"], compact = False)
az.plot_trace(idata, var_names = ["se2"], compact = False)
az.plot_trace(idata, var_names = ["se3"], compact = False)
az.plot_trace(idata, var_names = ["sc"], compact = False)

#Forest_plot
    #GoToTravel
az.plot_forest(idata, var_names = ["es1"], combined = False,
                colors = "C2", figsize = (11.5, 5), hdi_prob = 0.95, r_hat = False, quartiles = True,
                markersize = 6)
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.title(r"$\leftarrow$""increased COVID-19 death \n did not increase COVID-19 death"r"$\rightarrow$", fontsize = 16)
plt.savefig("/Users/tk/Documents/TMDU/COVID19/20230123/Figure/GoToTravel.png", dpi = 300)
    #GoToEat
az.plot_forest(idata, var_names = ["es2"], combined = False,
                colors = "C2", figsize = (11.5, 5), hdi_prob = 0.95, r_hat = False, quartiles = True,
                markersize = 10)
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.title(r"$\leftarrow$""increased COVID-19 death \n did not increase COVID-19 death"r"$\rightarrow$", fontsize = 16)
plt.savefig("/Users/tk/Documents/TMDU/COVID19/20230123/Figure/GoToEat.png", dpi = 300)
    #SE1
az.plot_forest(idata, var_names = ["se1"],
                combined = False, colors = "C1", figsize = (11.5, 5),
                hdi_prob = 0.95, r_hat = False)
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.title(r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$")
plt.savefig("/Users/tk/Documents/TMDU/COVID19/20230123/Figure/SE1.png", dpi = 300)
    #SE2
az.plot_forest(idata, var_names = ["se2"],
                combined = False, colors = "C1", figsize = (11.5, 5),
                hdi_prob = 0.95, r_hat = False)
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.title(r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$")
plt.savefig("/Users/tk/Documents/TMDU/COVID19/20230123/Figure/SE2.png", dpi = 300)
    #SE3
az.plot_forest(idata, var_names = ["se3"],
                combined = False, colors = "C1", figsize = (11.5, 5),
                hdi_prob = 0.95, r_hat = False)
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.title(r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$")
plt.savefig("/Users/tk/Documents/TMDU/COVID19/20230123/Figure/SE3.png", dpi = 300)
    #SC
az.plot_forest(idata, var_names = ["sc"],
                combined = False, colors = "C1", figsize = (11.5, 5),
                hdi_prob = 0.95, r_hat = False)
plt.axvline(x = 0, color = "red", linestyle = "dashed")
plt.title(r"$\leftarrow$""failed to reduce COVID-19 death \n reduced COVID-19 death"r"$\rightarrow$")
plt.savefig("/Users/tk/Documents/TMDU/COVID19/20230123/Figure/SC.png", dpi = 300)

#plot actual deaths and Byesian estimation graphs togeher.
import math
m = np.zeros(N)
mc = np.zeros(N)
for i in range(10):
    m += np.array(az.summary(idata2, var_names = ["r_e"], coords = {"r_e_dim_1":[i]})["mean"])
    mc += np.array(az.summary(idata2, var_names = ["r_c"], coords = {"r_c_dim_1":[i]})["mean"])
Fig_start_day = 50
GoTo_end_day = 347

df_10 = df.Hokkaido + df.Tokyo + df.Kanagawa + df.Saitama + df.Chiba + df.Aichi + df.Osaka + df.Hyogo + df.Fukuoka + df.Okinawa

sns.relplot(x = date[Fig_start_day:GoTo_end_day], y = df_10[Fig_start_day:GoTo_end_day], kind = "line", height = 10, aspect = 1.618, label = "actual mortality", color = "b")
plt.plot(date[Fig_start_day:GoTo_end_day], m[Fig_start_day:GoTo_end_day], label = "actual mortality rate")
plt.plot(date[Fig_start_day:GoTo_end_day], mc[Fig_start_day:GoTo_end_day], label = "counterfactual mortality rate")
plt.title("The 10 prefectures with the most deaths", fontsize = 30)
plt.ylabel("Number of death", fontsize = 25)
plt.xlabel("date", fontsize = 25)
plt.fill_between(x = [GoTo_start, GoTo_end], y1 = 0, y2 = max(df_10[Fig_start_day:GoTo_end_day]), color = "r", alpha = 0.1, label = "Go To Travel")
plt.fill_between(x = [GoToEat_start, GoTo_end], y1 = 0, y2 = max(df_10[Fig_start_day:GoTo_end_day]), color = "b", alpha = 0.1, label = "Go To Eat")
plt.subplots_adjust(top = 0.95, bottom = 0.05, left = 0.05)
plt.legend(fontsize = 25)
plt.savefig("/Users/tk/Documents/TMDU/COVID19/20230123/Figure/plot_without_Goto.png", dpi = 300)

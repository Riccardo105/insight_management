import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.stats import linregress

df = pd.read_csv("Mental_Health_and_Social_Media_balance_Dataset.csv")
df1 = pd.read_csv("smmh.csv")

Q1_sleep_quality = pd.read_csv("Q1_Sleep_Quality")
Q1_sleep_issues = pd.read_csv("Q1_Sleep_Issues")
Q2_validation_seeking = pd.read_csv("Q2_Validation_Seeking")
Q3_accademic_distraction = pd.read_csv("Q3_Accademic_Distraction")

def filter_df():

    # filtering out entries otuside of student age range
    age_column = "Age"
    min_age = 15
    max_age = 25

    df_filtered = df[(df[age_column] >= min_age) & (df[age_column] <= max_age)]

    return df_filtered

def fitler_df1():

    # filtering out entries outisde of occupation category
    allowed_occupation = ["University Student", "School Student"]
    df_filtered = df1[df1["4. Occupation Status"].isin(allowed_occupation)]

    return df_filtered



def Q1_sleep_quality_prep():

    #filtering age group
    df_filtered = filter_df()
   
    # renaming fields for clarity
    column_mapping = {
    "Age": "Age",
    "Daily_Screen_Time(hrs)": "Daily_Screen_Time",
    "Sleep_Quality(1-10)": "Sleep_Quality",}

    df_new = df_filtered[list(column_mapping.keys())].rename(columns=column_mapping)

    # Round screen time to nearest integer
    df_new["Daily_Screen_Time"] = df_new["Daily_Screen_Time"].round().astype(int)

    # Convert sleep quality to integer (drop .0)
    df_new["Sleep_Quality"] = df_new["Sleep_Quality"].astype(int)

    output_file = "Q1_Sleep_Quality"
    df_new.to_csv(output_file, index=False)


def Q1_sleep_issues_prep():

    df_filtered = fitler_df1()

    #renaming fields for clarity
    column_mapping = {
        "4. Occupation Status": "Occupation",
        "8. What is the average time you spend on social media every day?": "Daily_Screen_Time",
        "20. On a scale of 1 to 5, how often do you face issues regarding sleep?": "Frequency_of_Sleep_Issues"
    }
    df_new = df_filtered[list(column_mapping.keys())].rename(columns=column_mapping)

    # Map Daily_Screen_Time to numeric (using the range mid-point as reference)
    screen_time_mapping = {
        "Less than an Hour": 0.5,
        "Between 1 and 2 hours": 1.5,
        "Between 2 and 3 hours": 2.5,
        "Between 3 and 4 hours": 3.5,
        "Between 4 and 5 hours": 4.5,
        "More than 5 hours": 5.5
    }

    df_new["Daily_Screen_Time"] = df_new["Daily_Screen_Time"].map(screen_time_mapping)

    # Optional: export CSV
    output_file = "Q1_Sleep_Issues"
    df_new.to_csv(output_file, index=False)


def Q2_prep():
    
    df_filtered = fitler_df1()

    column_mapping = {
        "13. On a scale of 1 to 5, how much are you bothered by worries?": "Worries_Frequency",
        "18. How often do you feel depressed or down?": "Depression_Frequency",
        "15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?": "Comparison_Frequency",
        "17. How often do you look to seek validation from features of social media?": "Seeking_Validation_Frequency",
    }

    df_new = df_filtered[list(column_mapping.keys())].rename(columns=column_mapping)

    #group the entries by low/high level for independent t-test
    df_new["Worries_Level"] = df_new["Worries_Frequency"].apply(lambda x: "Low" if x in [1, 2] else "High")
    df_new["Depression_Level"] = df_new["Depression_Frequency"].apply(lambda x: "Low" if x in [1, 2] else "High")

    # Optional: export CSV
    output_file = "Q2_Validation_Seeking"
    df_new.to_csv(output_file, index=False)
    
def Q3_prep():

    df_filtered = fitler_df1()

    column_mapping = {
        "8. What is the average time you spend on social media every day?": "Daily_Screen_Time",
        "9. How often do you find yourself using Social media without a specific purpose?": "Frequency_of_Purposless_Use",
        "10. How often do you get distracted by Social media when you are busy doing something?": "Frequency_of_Distraction",
        "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?": "Interest_Flactuation"
    }

    df_new = df_filtered[list(column_mapping.keys())].rename(columns=column_mapping)

     # Map Daily_Screen_Time to numeric (using the range mid-point as reference)
    screen_time_mapping = {
        "Less than an Hour": 0.5,
        "Between 1 and 2 hours": 1.5,
        "Between 2 and 3 hours": 2.5,
        "Between 3 and 4 hours": 3.5,
        "Between 4 and 5 hours": 4.5,
        "More than 5 hours": 5.5
    }

    df_new["Daily_Screen_Time"] = df_new["Daily_Screen_Time"].map(screen_time_mapping)


    # mappig purposless use into two groups split into above and below mean
    median_purposeless = df_new['Frequency_of_Purposless_Use'].median()
    
  
    # Logic: if values is less than median: Low purposless use  and vice-versa
    df_new["Purposeless_Level"] = np.where(
        # Condition: If purposeless frequency is LESS than the median (i.e., less purposeless use)
        df_new['Frequency_of_Purposless_Use'] < median_purposeless, 
        "Low",   
        "High"
    )

    # Optional: export CSV
    output_file = "Q3_Accademic_Distraction"
    df_new.to_csv(output_file, index=False)




def Question1_quality():

    r, p = pearsonr(Q1_sleep_quality["Daily_Screen_Time"], Q1_sleep_quality["Sleep_Quality"])

    grouped = Q1_sleep_quality.groupby("Daily_Screen_Time")["Sleep_Quality"]
    means = grouped.mean()
    se = grouped.sem()

    plt.errorbar(means.index, means.values, yerr=se.values, fmt='o-', capsize=5)
    plt.xlabel('Average Social Media Time (hrs)')
    plt.ylabel('Sleep Quality (Self-Reported)')
    plt.title('Social Media Use vs. Sleep Quality')

    text = f"r = {r:.5f}\np = {p:.5f}"
    plt.text(
    0.05, 0.05,         
    text,
    transform=plt.gca().transAxes,  
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", alpha=0.2)
)
    plt.show()

def Question1_issues():

    r, p = pearsonr(Q1_sleep_issues["Daily_Screen_Time"], Q1_sleep_issues["Frequency_of_Sleep_Issues"])

    grouped = Q1_sleep_issues.groupby("Daily_Screen_Time")["Frequency_of_Sleep_Issues"]
    means = grouped.mean()
    se = grouped.sem()

    plt.errorbar(means.index, means.values, yerr=se.values, fmt='o-', capsize=5)
    plt.xlabel('Average Social Media Time (hrs)')
    plt.ylabel('Frequency of Sleep Issues (Self-Reported)')
    plt.title('Social Media Use vs. Sleep Quality')

    text = f"r = {r:.5f}\np = {p:.5f}"
    plt.text(
    0.1, 0.05,         
    text,
    transform=plt.gca().transAxes,  
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", alpha=0.2)
)
    plt.show()

def Question2_depression_validation():

    low_depression_group = Q2_validation_seeking[Q2_validation_seeking["Depression_Level"] == "Low"]["Seeking_Validation_Frequency"]
    high_depression_group = Q2_validation_seeking[Q2_validation_seeking["Depression_Level"] == "High"]["Seeking_Validation_Frequency"]

    t, p = ttest_ind(low_depression_group, high_depression_group, equal_var=False)
    

    means = [low_depression_group.mean(), high_depression_group.mean()]
    se = [low_depression_group.sem(), high_depression_group.sem()]
    groups = ["Low Depression", "High Depression"]

    plt.bar(groups, means, yerr=se, capsize=5, color=['teal', 'orange'])
    plt.ylabel('Mean Social Media Validation Score')
    plt.title('Validation-seeking by Depression Group')

    text = f"r = {t:.5f}\np = {p:.5f}"
    plt.text(
    0.03, 0.95,         
    text,
    transform=plt.gca().transAxes,  
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.show()

def Question2_depression_comparison():

    low_depression_group = Q2_validation_seeking[Q2_validation_seeking["Depression_Level"] == "Low"]["Comparison_Frequency"]
    high_depression_group = Q2_validation_seeking[Q2_validation_seeking["Depression_Level"] == "High"]["Comparison_Frequency"]

    t, p = ttest_ind(low_depression_group, high_depression_group, equal_var=False)
    

    means = [low_depression_group.mean(), high_depression_group.mean()]
    se = [low_depression_group.sem(), high_depression_group.sem()]
    groups = ["Low Depression", "High Depression"]

    plt.bar(groups, means, yerr=se, capsize=5, color=['teal', 'orange'])
    plt.ylabel('Mean Social Media Comparison Score')
    plt.title('Comparison by Depression Group')

    text = f"r = {t:.5f}\np = {p:.5f}"
    plt.text(
    0.03, 0.95,         
    text,
    transform=plt.gca().transAxes,  
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.show()

def Question2_worries_validation():

    low_Worries_group = Q2_validation_seeking[Q2_validation_seeking["Worries_Level"] == "Low"]["Seeking_Validation_Frequency"]
    high_Worries_group = Q2_validation_seeking[Q2_validation_seeking["Worries_Level"] == "High"]["Seeking_Validation_Frequency"]

    t, p = ttest_ind(low_Worries_group, high_Worries_group, equal_var=False)
    

    means = [low_Worries_group.mean(), high_Worries_group.mean()]
    se = [low_Worries_group.sem(), high_Worries_group.sem()]
    groups = ["Low worries", "High worries"]

    plt.bar(groups, means, yerr=se, capsize=5, color=['teal', 'orange'])
    plt.ylabel('Mean Social Media Validation Score')
    plt.title('Validation-seeking by Worries Group')

    text = f"r = {t:.5f}\np = {p:.5f}"
    plt.text(
    0.03, 0.95,         
    text,
    transform=plt.gca().transAxes,  
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.show()

def Question2_worries_comparison():

    low_Worries_group = Q2_validation_seeking[Q2_validation_seeking["Worries_Level"] == "Low"]["Comparison_Frequency"]
    high_Worries_group = Q2_validation_seeking[Q2_validation_seeking["Worries_Level"] == "High"]["Comparison_Frequency"]

    t, p = ttest_ind(low_Worries_group, high_Worries_group, equal_var=False)
    

    means = [low_Worries_group.mean(), high_Worries_group.mean()]
    se = [low_Worries_group.sem(), high_Worries_group.sem()]
    groups = ["Low worries", "High worries"]

    plt.bar(groups, means, yerr=se, capsize=5, color=['teal', 'orange'])
    plt.ylabel('Mean Social Media Comparison Score')
    plt.title('Comparison by Worries Group')

    text = f"r = {t:.5f}\np = {p:.5f}"
    plt.text(
    0.03, 0.95,         
    text,
    transform=plt.gca().transAxes,  
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.show()

def Question3_Distraction ():
    # Filter the data into the two groups
    high_purposeless_df = Q3_accademic_distraction[Q3_accademic_distraction['Purposeless_Level'] == 'High']
    low_purposeless_df = Q3_accademic_distraction[Q3_accademic_distraction['Purposeless_Level'] == 'Low']

    # Line for High Purposelessness Group (Blue)
    x_high = high_purposeless_df['Daily_Screen_Time']
    y_high = high_purposeless_df['Frequency_of_Distraction']
    if len(x_high) > 1:
        slope_h, intercept_h, r_val_h, p_val_h, std_err_h = linregress(x_high, y_high)
        plt.plot(x_high, slope_h * x_high + intercept_h, color='blue', linestyle='-', linewidth=2)

    # Line for Low Purposelessness Group (Red)
    x_low = low_purposeless_df['Daily_Screen_Time']
    y_low = low_purposeless_df['Frequency_of_Distraction']
    if len(x_low) > 1:
        slope_l, intercept_l, r_val_l, p_val_l, std_err_l = linregress(x_low, y_low)
        plt.plot(x_low, slope_l * x_low + intercept_l, color='red', linestyle='-', linewidth=2)

    # ---  STATS FOR BLUE LINE (High Purposeless) ---
    text_h = f"High Purp.: \nr = {r_val_h:.5f}\np = {p_val_h:.5f}\nstd_err = {std_err_h:.5f}"
    plt.text(
        4.26, 3.55,  
        text_h,
        transform=plt.gca().transData,
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.5", fc="lightblue", alpha=0.6)
    )

    # --- STATS FOR RED LINE (Low Purposeless) ---
    text_l = f"Low Purp.: \nr = {r_val_l:.3f}\np = {p_val_l:.5f}\nstd_err={std_err_l:.5f}"
    plt.text(
        4.26, 2.5,  
        text_l,
        transform=plt.gca().transData,
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.5", fc="lightcoral", alpha=0.6)
    )

    plt.xlabel('Total Daily Screen Time')
    plt.ylabel('Frequency of Academic Distraction')
    plt.title('Academic Distraction by Screen Time and Purpose Level')

    
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()
    

def Question3_Interest ():
    # 1. Filter the data into the two groups
    high_purposeless_df = Q3_accademic_distraction[Q3_accademic_distraction['Purposeless_Level'] == 'High']
    low_purposeless_df = Q3_accademic_distraction[Q3_accademic_distraction['Purposeless_Level'] == 'Low']

    # Line for High Purposelessness Group (Blue)
    x_high = high_purposeless_df['Daily_Screen_Time']
    y_high = high_purposeless_df['Interest_Flactuation']
    # Ensure there are enough data points (at least 2) for linear regression
    if len(x_high) > 1:
        slope_h, intercept_h, r_val_h, p_val_h, std_err_h = linregress(x_high, y_high)
        plt.plot(x_high, slope_h * x_high + intercept_h, color='blue', linestyle='-', linewidth=1)

    # Line for Low Purposelessness Group (Red)
    x_low = low_purposeless_df['Daily_Screen_Time']
    y_low = low_purposeless_df['Interest_Flactuation']
    if len(x_low) > 1:
        slope_l, intercept_l, r_val_l, p_val_l, std_err_l = linregress(x_low, y_low)
        plt.plot(x_low, slope_l * x_low + intercept_l, color='red', linestyle='-', linewidth=2)


    # ---  STATS FOR BLUE LINE (High Purposeless) ---
    text_h = f"High Purp.: \nr = {r_val_h:.5f}\np = {p_val_h:.5f}\nstd_err = {std_err_h:.5f}"
    plt.text(
        4.26, 3.3,  
        text_h,
        transform=plt.gca().transData,
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.5", fc="lightblue", alpha=0.6)
    )

    # --- STATS FOR RED LINE (Low Purposeless) ---
    text_l = f"Low Purp.: \nr = {r_val_l:.3f}\np = {p_val_l:.5f}\nstd_err={std_err_l:.5f}"
    plt.text(
        4.26, 2.8,  
        text_l,
        transform=plt.gca().transData,
        fontsize=9,
        bbox=dict(boxstyle="round, pad=0.5", fc="lightcoral", alpha=0.6)
    )

    plt.xlabel('Total Daily Screen Time')
    plt.ylabel('Flactuation of interest in daily acitivities')
    plt.title('Loss of interest by Screen Time and Purpose Level')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def Question3_Distraction_Mediation():

    X_var = 'Daily_Screen_Time'  # Independent variable X
    M_var = 'Frequency_of_Purposless_Use' # Mediator M
    Y_var = 'Frequency_of_Distraction'    # Dependent Variable Y

    mediation_results = pg.mediation_analysis(
    data=Q3_accademic_distraction, 
    x=X_var,
    m=M_var,
    y=Y_var,
    n_boot=5000, 
    seed=42 
)

    print(mediation_results)

def Question3_Interest_Mediation():
   
    X_var = 'Daily_Screen_Time'  # Independent variable X
    M_var = 'Frequency_of_Purposless_Use' # Mediator M
    Y_var = 'Interest_Flactuation'    # Dependent Variable Y

    mediation_results = pg.mediation_analysis(
    data=Q3_accademic_distraction, 
    x=X_var,
    m=M_var,
    y=Y_var,
    n_boot=5000, 
    seed=42 
)

    print(mediation_results)
    


#Question1_quality()
#Question2_worries_comparison()
Question3_Distraction_Mediation()
Question3_Interest_Mediation()




import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

df = pd.read_csv("Mental_Health_and_Social_Media_balance_Dataset.csv")
df1 = pd.read_csv("smmh.csv")




def stress_by_age():
    mean_stress_by_age = df.groupby("Age")["Stress_Level(1-10)"].mean().reset_index()
    plt.bar(mean_stress_by_age['Age'], mean_stress_by_age['Stress_Level(1-10)'])
    plt.xlabel('Age')
    plt.ylabel('Average Stress Level')
    plt.title('Mean Stress Level by Age')
    plt.show()

   


def hrs_screen_time():
    hrs_less_1 = df[(df["Daily_Screen_Time(hrs)"] <= 1)].shape[0]
    hrs_1_2 = df[(df["Daily_Screen_Time(hrs)"] >= 1) & (df["Daily_Screen_Time(hrs)"] <= 2)].shape[0]
    hrs_3_4 = df[(df["Daily_Screen_Time(hrs)"] >= 3) & (df["Daily_Screen_Time(hrs)"] <= 4)].shape[0]
    hrs_5_6 = df[(df["Daily_Screen_Time(hrs)"] >= 5) & (df["Daily_Screen_Time(hrs)"] <= 6)].shape[0]
    hrs_7 = df[(df["Daily_Screen_Time(hrs)"] >= 7)].shape[0]

    counts = [hrs_less_1, hrs_1_2, hrs_3_4, hrs_5_6, hrs_7]
    hrs_labels = ["-1", "1-2", "3-4", "5-6", "7"]

    plt.bar(hrs_labels, counts)
    plt.xlabel("Daily Screen Time")
    plt.ylabel("Count")
    plt.title("count of entries by daily screen time range")
    plt.show()



def Question1():

    r, p = pearsonr(df["Daily_Screen_Time(hrs)"], df["Sleep_Quality(1-10)"])
    print(f'Correlation coefficient (r): {r}, p-value: {p}')

    grouped = df.groupby("Sleep_Quality(1-10)")["Daily_Screen_Time(hrs)"]
    means = grouped.mean()
    se = grouped.sem()

    plt.errorbar(means.index, means.values, yerr=se.values, fmt='o-', capsize=5)
    plt.xlabel('Average Social Media Time (hrs)')
    plt.ylabel('Sleep Quality (Self-Reported)')
    plt.title('Social Media Use vs. Sleep Quality')
    plt.show()

def Question2():

    low_depression_group = df1[df1["18. How often do you feel depressed or down?"] <= 2]["17. How often do you look to seek validation from features of social media?"]
    high_depression_group = df1[df1["18. How often do you feel depressed or down?"] > 2]["17. How often do you look to seek validation from features of social media?"]

    t_stat, p_val = ttest_ind(low_depression_group, high_depression_group, equal_var=False)
    print(f"t = {t_stat:.2f}, p = {p_val:f}")

    means = [low_depression_group.mean(), high_depression_group.mean()]
    se = [low_depression_group.sem(), high_depression_group.sem()]
    groups = ["Low Depression", "High Depression"]

    plt.bar(groups, means, yerr=se, capsize=5, color=['teal', 'orange'])
    plt.ylabel('Mean Social Media Validation Score')
    plt.title('Validation-seeking by Depression Group')
    plt.show()

    

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

    # formatting daily screen time to Nominal
    bins = [0, 2, 4, float("inf")]  # 0–2, 3–4, 5+
    labels = [
        "Between 1 and 2 hours",
        "Between 3 and 4 hours",
        "More than 5 hours"
    ]

    df_new["Daily_Screen_Time"] = pd.cut(
    df_new["Daily_Screen_Time"],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True
)

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

    # Optional: export CSV
    output_file = "Q3_Accademinc_Distraction"
    df_new.to_csv(output_file, index=False)



Q3_prep()



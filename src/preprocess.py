import json
import os as os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_model import Bug


def get_project_root() -> Path:
    return Path(__file__).parent.parent


ROOT_DIR = get_project_root()
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
DATASET_DIR = os.path.join(DATASET_DIR, "data")


# resource of this file: https://github.com/EhsanMashhadi/ISSRE2023-BugSeverityPrediction

def categorical_to_number_d4j(df):
    df['Severity'].replace("Critical", 0, inplace=True)
    df['Severity'].replace("High", 1, inplace=True)
    df['Severity'].replace("Medium", 2, inplace=True)
    df['Severity'].replace("Low", 3, inplace=True)

    return df


def categorical_to_number_bugsjar(df):
    df['Severity'].replace("Blocker", 0, inplace=True)
    df['Severity'].replace("Critical", 0, inplace=True)
    df['Severity'].replace("Major", 1, inplace=True)
    df['Severity'].replace("Minor", 3, inplace=True)
    df['Severity'].replace("Trivial", 3, inplace=True)
    return df


def split_dataset():
    d4j = pd.read_csv(os.path.join(DATASET_DIR, "d4j_methods_sc_metrics_comments.csv"))
    new_d4j = categorical_to_number_d4j(d4j)
    bugs_jar = pd.read_csv(os.path.join(DATASET_DIR, "bugsjar_methods_sc_metrics_comments.csv"))
    new_bugs_jar = categorical_to_number_bugsjar(bugs_jar)
    bugs = []
    bugs.extend(create_bugs(new_d4j))
    bugs.extend(create_bugs(new_bugs_jar))

    df_bugs = pd.DataFrame(bugs)
    df_bugs.drop_duplicates(keep='first', inplace=True)

    train, test = train_test_split(df_bugs, test_size=0.15, random_state=666, shuffle=True)
    train, val = train_test_split(train, test_size=0.15, random_state=666, shuffle=True)

    cols = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']

    write_bugs(train, "train")
    write_bugs(val, "valid")
    write_bugs(test, "test")


def create_bugs(df):
    bugs = []
    for index, row in df.iterrows():
        if row["IsBuggy"]:
            bug = Bug(project_name=row["ProjectName"], project_version=row["ProjectVersion"], severity=row["Severity"],
                      code=row["SourceCode"], code_comment=row["CodeComment"],
                      code_no_comment=row["CodeNoComment"], lc=row["LC"], pi=row["PI"], ma=row["MA"],
                      nbd=row["NBD"], ml=row["ML"], d=row["D"], mi=row["MI"], fo=row["FO"], r=row["R"], e=row["E"])
            bugs.append(bug.__dict__)
    return bugs


def write_bugs(bugs, name):
    with open("{}.jsonl".format(name), 'w') as f:
        for bug in bugs.to_dict("records"):
            f.write(json.dumps(bug) + "\n")
    df = pd.DataFrame(bugs)
    df.to_csv("{}.csv".format(name), index=False)


if __name__ == '__main__':
    split_dataset()

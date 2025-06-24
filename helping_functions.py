

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import networkx as nx

import google.generativeai as genai
from google.api_core import retry

import os
import time

"""a collection of the helping functions that are used by the regulation analysis process"""


def print_communities_results(communities, regulations_DataFrame:pd.DataFrame, 
                              max_nodes_to_print:int=20, Aggregate_sub_Controls:bool=True, **kwargs)->None:
    """
    Print the results of community detection in a regulatory framework.

    This function prints the number of communities detected and details about each community,
    including the number of nodes, family names, and optionally the full list of nodes.

    Parameters:
    communities (list): A list of communities, where each community is a collection of nodes.
    regulations_DataFrame (pd.DataFrame): A DataFrame containing regulatory information.
    max_nodes_to_print (int, optional): Maximum number of nodes to print for each community. Defaults to 20.
    Aggregate_sub_Controls (bool, optional): If True, aggregates sub-controls under main controls. Defaults to True.
    **kwargs: Additional keyword arguments. Supports 'k' or 'num_communities' to specify the number of communities.

    Returns:
    None: This function doesn't return any value, it only prints the results.
    """
    # print how many communities there are
    if 'k' in kwargs.keys():
        print(f"there are {kwargs['k']} Communities")
    elif 'num_communities' in kwargs.keys():
        print(f"there are {kwargs['num_communities']} Communities")
    else:
        try:
            print(f"there are {len(communities)} Communities")
        except Exception as e: 
            print(e)

    # print each Community
    for i,Community in enumerate(communities):
        print(f"\nCommunity {i} if of length {len(Community)}\nand contains:")
        if Aggregate_sub_Controls:
            print(regulations_DataFrame.loc[regulations_DataFrame["Control Identifier"].isin(Community)].drop_duplicates(["Main Control Name","Family Name"]).value_counts(["Family Name"]))
        else:
            print(regulations_DataFrame.loc[regulations_DataFrame["Control Identifier"].isin(Community)].value_counts(["Family Name"]))
        if len(Community) <= max_nodes_to_print:
            print(Community)

    return



def write_communities_in_regulations(communities, regulations_DataFrame:pd.DataFrame, **kwargs)->pd.DataFrame:
    """
    Write community assignments to the regulations DataFrame.

    This function assigns community labels to regulations based on the provided community structure.

    Parameters:
    communities (list): A list of communities, where each community is a collection of nodes.
    regulations_DataFrame (pd.DataFrame): A DataFrame containing regulatory information.
    **kwargs: Additional keyword arguments.
        communities_name (str): The name of the column to store community assignments. Defaults to "new communities".
        node_column (str): The name of the column in regulations_DataFrame that contains node identifiers. Defaults to "Control Identifier".

    Returns:
    pd.DataFrame: The updated regulations DataFrame with community assignments added.
    """
    if "communities_name" not in kwargs.keys():
        communities_name = "new communities"
    else:
        communities_name = kwargs["communities_name"]
    if "node_column" not in kwargs.keys():
        node_column = "Control Identifier"
    else:
        node_column = kwargs["node_column"]

    for i,Community in enumerate(communities):
        # regulations_DataFrame.loc[Community,communities_name] = i
        regulations_DataFrame.loc[regulations_DataFrame[node_column].isin(Community),communities_name] = i

    return regulations_DataFrame


def print_and_write_communities_results(communities, regulations_DataFrame:pd.DataFrame, 
                                        max_nodes_to_print:int=20, Aggregate_sub_Controls:bool=True, **kwargs)->pd.DataFrame:
    """
    Print community detection results and write them to the regulations DataFrame.

    This function prints information about detected communities, including their number and composition,
    and updates the regulations DataFrame with community assignments.

    Parameters:
    communities (list): A list of communities, where each community is a collection of nodes.
    regulations_DataFrame (pd.DataFrame): A DataFrame containing regulatory information.
    max_nodes_to_print (int, optional): Maximum number of nodes to print for each community. Defaults to 20.
    Aggregate_sub_Controls (bool, optional): If True, aggregates sub-controls under main controls when printing. Defaults to True.
    **kwargs: Additional keyword arguments.
        communities_name (str): Name of the column to store community assignments. Defaults to "new communities".
        node_column (str): Name of the column in regulations_DataFrame that contains node identifiers. Defaults to "Control Identifier".
        k (int): Number of communities (alternative to num_communities).
        num_communities (int): Number of communities (alternative to k).

    Returns:
    pd.DataFrame: The updated regulations DataFrame with community assignments added.
    """
    
    if "communities_name" not in kwargs.keys():
        communities_name = "new communities"
    else:
        communities_name = kwargs["communities_name"]
    if "node_column" not in kwargs.keys():
        node_column = "Control Identifier"
    else:
        node_column = kwargs["node_column"]

    # print how many communities there are
    if 'k' in kwargs.keys():
        print(f"there are {kwargs['k']} Communities")
    elif 'num_communities' in kwargs.keys():
        print(f"there are {kwargs['num_communities']} Communities")
    else:
        try:
            print(f"there are {len(communities)} Communities")
        except Exception as e: 
            print(e)
    
    # print each Community
    for i,Community in enumerate(communities):
        # write the new Community
        regulations_DataFrame.loc[regulations_DataFrame[node_column].isin(Community),communities_name] = i

        # print the new Community
        print(f"\nCommunity {i} if of length {len(Community)}\nand contains:")
        if Aggregate_sub_Controls:
            print(regulations_DataFrame.loc[regulations_DataFrame["Control Identifier"].isin(Community)].drop_duplicates(["Main Control Name","Family Name"]).value_counts(["Family Name"]))
        else:
            print(regulations_DataFrame.loc[regulations_DataFrame["Control Identifier"].isin(Community)].value_counts(["Family Name"]))
        if len(Community) <= max_nodes_to_print:
            print(Community)
    return regulations_DataFrame

def add_summarization(regulations_DataFrame:pd.DataFrame, relations_DataFrame:pd.DataFrame, is_Main_Controls_Only:bool=True) -> pd.DataFrame:
    """
    Summarize cyber regulations groups using AI and add the summaries to the relations DataFrame.

    This function processes each group of regulations, generates a summary using an AI model,
    and adds the summary to the relations DataFrame.

    Parameters:
    regulations_DataFrame (pd.DataFrame): DataFrame containing detailed regulation information.
    relations_DataFrame (pd.DataFrame): DataFrame containing the groups of the cyber regulations that where found with other analysis.
    is_Main_Controls_Only (bool, optional): If True, uses main control names; otherwise, uses control identifiers. Defaults to True.

    Returns:
    pd.DataFrame: Updated relations_DataFrame with added summaries for each group of regulations.
    """
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    for inx in relations_DataFrame.index:
        # get the relevant regulations
        if is_Main_Controls_Only:
            group = regulations_DataFrame["Main Control Name"].isin(relations_DataFrame.loc[inx,"names of regulations"])
        else:
            group = regulations_DataFrame["Control Identifier"].isin(relations_DataFrame.loc[inx,"names of regulations"])
        regulations_examples:str = ""
        for i,regulations_text in enumerate(regulations_DataFrame.loc[group,"Full Text"]):
            regulations_examples += f"Regulation {i+1}:\n{regulations_text}\n\n"

            # break
        # print(regulations_examples)

        # summarize the regulations
        system_instruction:str = "You are an expert in cyber regulations. \nYour task is to read all the regulations you are given and summarize them in up to 5 short and concise bullet points. \nIn your answers give the general topic that represents all of the regulations you were given."
        # model = genai.GenerativeModel(model_name="gemini-2.0-flash",system_instruction=system_instruction)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp",system_instruction=system_instruction)
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        response = model.generate_content(f"Please summarize this group of cyber regulations:\n\n{regulations_examples}",
                                          request_options=retry_policy,)

        # add the responses to the summary column in the relations_DataFrame
        relations_DataFrame.loc[inx,"Summary"] = response.text.strip()

        # print the response
        # print(f"Response for group {inx}:\n{response}")
        print(f"Response for group {inx}\ncontaining the regulations {relations_DataFrame.loc[inx,'names of regulations']}:\n\n{response.text.strip()}")
        print("\n"+"-"*100+"\n")

        time.sleep(6) # sleep for a little bit because of RPM limitations

    return relations_DataFrame



def get_mutual_relations(regulations_DataFrame:pd.DataFrame, relations_column:str|list[str], relations_graph:nx.Graph|nx.DiGraph=None,
                         name_column:str = "Main Control Name", family_column:str = "Family Name", is_Main_Controls_Only:bool=True, 
                         do_print:bool = True)->tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.DataFrame]:
    """
        Analyze mutual relations between control groups in a regulatory framework and compute various statistics.

        This function processes a DataFrame of regulatory controls, identifies mutual and non-mutual relations
        between control groups, and calculates various statistics about these relations, including in-family and
        out-of-family connections.

        Parameters:
        regulations_DataFrame (pd.DataFrame): DataFrame containing regulatory control data.
        relations_column (str | list[str]): Column name(s) in the DataFrame that contain relation information.
        relations_graph (nx.Graph | nx.DiGraph, optional): A NetworkX graph object representing the relations. Default is None.
        name_column (str): Column name for the control names. Default is "Main Control Name".
        family_column (str): Column name for the control family names. Default is "Family Name".
        is_Main_Controls_Only (bool): If True, only consider main controls. Default is True.
        do_print (bool): If True, print progress information. Default is True.

        Returns:
        tuple: A tuple containing four elements:
            - mutually_related (pd.DataFrame): DataFrame of mutually related controls.
            - not_mutually_related (pd.DataFrame): DataFrame of non-mutually related controls.
            - in_family_connection_statistics (pd.Series): Series containing various statistics about in-family connections.
            - relations_statistics (pd.DataFrame): DataFrame containing detailed statistics about relations for each family.
        """
    # set default for is_Main_Controls_Only
    if name_column == "Main Control Name":
        is_Main_Controls_Only:bool=True
    elif name_column == "Control Identifier":
        is_Main_Controls_Only:bool=False
    else:
        print(f"unregular control column name: {name_column}")
    
    # make mutually_related and not_mutually_related DataFrames
    mutually_related = pd.DataFrame(columns=["origen control","related control","same family",])
    m_r = 0
    not_mutually_related = pd.DataFrame(columns=["origen control","related control","same family",])
    n_m_r = 0

    # group controls and sub-controls
    controls_and_subs = regulations_DataFrame.groupby(by=[name_column])

    for inxs,group in controls_and_subs:
        # print(group)
        if do_print: print(inxs)

        Control_name = inxs[0]
        # get Related_Controls for the whole group
        Related_Controls = set()
        for j, row in group.iterrows():
            if type(relations_column) == str:
                R = row[relations_column].keys()
                if is_Main_Controls_Only:
                    R = [Control[:Control.find('(')] if Control.find('(') != -1 else Control for Control in R]
                if "None" not in R and len(R) > 0:
                    Related_Controls = Related_Controls | set(R)
            elif type(relations_column) == list:
                for relation in relations_column:
                    R = row[relation].keys()
                    if is_Main_Controls_Only:
                        R = [Control[:Control.find('(')] if Control.find('(') != -1 else Control for Control in R]
                    if "None" not in R and len(R) > 0:
                        Related_Controls = Related_Controls | set(R)

        # for each of the Related_Controls check if relates back to the origen control
        for R_Control in Related_Controls:
            # print(R_Control)
            # if R_Control == "AU-15":
            #     R_Control = "AU-5"
            R_group = controls_and_subs.get_group(name=(R_Control,),)
            flag = False
            for k, R_row in R_group.iterrows():
                if type(relations_column) == str:
                    if is_Main_Controls_Only:
                        keys = [Control[:Control.find('(')] if Control.find('(') != -1 else Control for Control in R_row[relations_column].keys()]
                    else:
                        keys = R_row[relations_column].keys()
                    if Control_name in keys:
                        flag = True
                elif type(relations_column) == list:
                    for relation in relations_column:
                        if is_Main_Controls_Only:
                            keys = [Control[:Control.find('(')] if Control.find('(') != -1 else Control for Control in R_row[relation].keys()]
                        else:
                            keys = R_row[relation].keys()
                        if Control_name in keys:
                            flag = True
                            break
            if not flag:
                if do_print: print(f"{Control_name} is related to {R_Control} but not otherwise")
                same_family = regulations_DataFrame.loc[regulations_DataFrame[name_column]==Control_name,family_column].iloc[0] == regulations_DataFrame.loc[regulations_DataFrame[name_column]==R_Control,family_column].iloc[0]
                not_mutually_related.loc[n_m_r,:] = [Control_name,R_Control,same_family]
                # not_mutually_related.loc[n_m_r,:] = [Control_name,R_Control,Control_name[:2] == R_Control[:2]]
                n_m_r += 1
            else:
                # print(f"{Control_name} is mutually related to {R_Control}")
                same_family = regulations_DataFrame.loc[regulations_DataFrame[name_column]==Control_name,family_column].iloc[0] == regulations_DataFrame.loc[regulations_DataFrame[name_column]==R_Control,family_column].iloc[0]
                mutually_related.loc[m_r,:] = [Control_name,R_Control,same_family]
                # mutually_related.loc[m_r,:] = [Control_name,R_Control,Control_name[:2] == R_Control[:2]]
                m_r += 1

    # drop the mutual self connections 
    mutually_related = mutually_related.loc[mutually_related["origen control"] != mutually_related["related control"]]

    # in family connection statistics
    in_family_connection_statistics = pd.Series()
    families = regulations_DataFrame.groupby(by=[family_column]) # pre-group families 
    in_family_connection_statistics["number of families"] = regulations_DataFrame[family_column].nunique()
    # if is_Main_Controls_Only:
    #     num_Controls_in_family = families[name_column].nunique()
    # else:
    #     num_Controls_in_family = families.apply(len,include_groups=False)
    num_Controls_in_family = families[name_column].nunique()
    in_family_connection_statistics["mean number of controls in family"] = num_Controls_in_family.mean()
    in_family_connection_statistics["median number of controls in family"] = num_Controls_in_family.median()
    in_family_connection_statistics["std of number of controls in family"] = num_Controls_in_family.std()
    in_family_connection_statistics["Inter-quartile range of number of controls in family"] = num_Controls_in_family.quantile(0.75) - num_Controls_in_family.quantile(0.25)
    in_family_connection_statistics["number of participating controls"] = regulations_DataFrame.loc[:,[name_column,family_column]].dropna().nunique()[name_column]
    in_family_connection_statistics["balance score 1 (mean/std)(higher better)"] = in_family_connection_statistics["mean number of controls in family"] / in_family_connection_statistics["std of number of controls in family"]
    in_family_connection_statistics["balance score 2 (median/Inter-quartile range)(higher better)"] = in_family_connection_statistics["median number of controls in family"] / in_family_connection_statistics["Inter-quartile range of number of controls in family"]

    in_family_connection_statistics["same family connections"] = sum(not_mutually_related["same family"]==True) + sum(mutually_related["same family"]==True)
    in_family_connection_statistics["out of family connections"] = sum(not_mutually_related["same family"]==False) + sum(mutually_related["same family"]==False)
    
    # # ratio
    # in_family_connection_ratio_1 = (sum(not_mutually_related["same family"]==True) + sum(mutually_related["same family"]==True)/1) / (sum(not_mutually_related["same family"]==False) + sum(mutually_related["same family"]==False)/1)
    # in_family_connection_statistics["empiric connection ratio 1 (total/total)"] = in_family_connection_ratio_1
    # probability
    in_family_connection_statistics["empiric connection probability 1 (same family/total)(higher better)"] = (sum(not_mutually_related["same family"]==True) + sum(mutually_related["same family"]==True)/1) / (len(not_mutually_related) + len(mutually_related))

    # group families

    regulations_DataFrame
    num_Controls = regulations_DataFrame[name_column].nunique()
    # print(f"num_Controls = {num_Controls}")
    num_Family_Controls = families[name_column].nunique()
    names_Family_Controls = families[name_column].unique()
    sum_Controls = num_Family_Controls.sum()
    # print(f"sum_Controls = {sum_Controls}")
    # print(f"num_Family_Controls:\n{num_Family_Controls}")
    # print(f"names_Family_Controls:\n{names_Family_Controls}")

    # regulations_DataFrame["Family Name"].nunique()

    # calculate how many inside and outside relations each family have/can have, and statistics on the amount of in-family/out of family connections in relation to the number of total connections
    relations_statistics = pd.DataFrame(index=num_Family_Controls.index,
                                        columns=["number of regulations","names of regulations", 
                                                 "possible inside relations","actual inside relations","percentage inside relations",
                                                 "possible outside relations","actual outside relations","percentage outside relations",
                                                #  "ratio inner to outer relations",
                                                 "inner connections probability",
                                                 ])
    relations_statistics["number of regulations"] = num_Family_Controls
    relations_statistics["names of regulations"] = names_Family_Controls
    # print(f"relations_statistics:\n{relations_statistics}")
    # possible_inside_relations = pd.Series(0,index=num_Family_Controls.index)
    # possible_outside_relations = pd.Series(0,index=num_Family_Controls.index)

    # calculate how many inside and outside relations each family can have

    # get possible and actual number of relations
    for inx1 in num_Family_Controls.index:
        # get possible
        relations_statistics.loc[inx1,"possible inside relations"] = num_Family_Controls.loc[inx1] * (num_Family_Controls.loc[inx1] - 1)
        # possible_inside_relations.loc[inx1] = num_Family_Controls.loc[inx1] * (num_Family_Controls.loc[inx1] - 1)
        relations_statistics.loc[inx1,"possible outside relations"] = num_Family_Controls.loc[inx1] * (sum_Controls - num_Family_Controls.loc[inx1])
        # possible_outside_relations.loc[inx1] = num_Family_Controls.loc[inx1] * (sum_Controls - num_Family_Controls.loc[inx1])
        
        # get actual
        relations_statistics.loc[inx1,"actual inside relations"] = sum((not_mutually_related["origen control"].isin(names_Family_Controls[inx1])) & (not_mutually_related["same family"] == True))\
                                                                    + sum((mutually_related["origen control"].isin(names_Family_Controls[inx1])) & (mutually_related["same family"] == True))
        relations_statistics.loc[inx1,"actual outside relations"] = sum((not_mutually_related["origen control"].isin(names_Family_Controls[inx1])) & (not_mutually_related["same family"] == False))\
                                                                    + sum((mutually_related["origen control"].isin(names_Family_Controls[inx1])) & (mutually_related["same family"] == False))
        # pass
    
    # # calculate inner to outer relations ratio
    # try:
    #     relations_statistics["ratio inner to outer relations"] = relations_statistics["actual inside relations"] / relations_statistics["actual outside relations"]
    # except ZeroDivisionError:
    #     relations_statistics["ratio inner to outer relations"] = relations_statistics["actual inside relations"] / (relations_statistics["actual outside relations"] + 0.1)
    #     print("Error: Division by zero occurred.")
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    #     relations_statistics["ratio inner to outer relations"] = np.nan
    
    # calculate inner connections probability
    # print(f"actual inside relations = {relations_statistics['actual inside relations']}")
    # print(f"total relations = {(relations_statistics['actual inside relations'] + relations_statistics['actual outside relations'])}")
    try:
        relations_statistics["inner connections probability"] = relations_statistics["actual inside relations"] / (relations_statistics["actual inside relations"] + relations_statistics["actual outside relations"])
    except ZeroDivisionError:
        for inx1 in relations_statistics.index:
            if relations_statistics.loc[inx1,"actual inside relations"] == 0 or (relations_statistics.loc[inx1,"actual inside relations"] + relations_statistics.loc[inx1,"actual outside relations"]) == 0:
                relations_statistics.loc[inx1,"inner connections probability"] = 0
            else:
                relations_statistics.loc[inx1,"inner connections probability"] = relations_statistics.loc[inx1,"actual inside relations"] / (relations_statistics.loc[inx1,"actual inside relations"] + relations_statistics.loc[inx1,"actual outside relations"])
        print("Error: Division by zero occurred.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        relations_statistics["ratio inner to outer relations"] = np.nan


    # calculate percentages
    relations_statistics["percentage inside relations"] = relations_statistics["actual inside relations"] / relations_statistics["possible inside relations"]
    relations_statistics["percentage outside relations"] = relations_statistics["actual outside relations"] / relations_statistics["possible outside relations"]

    # print(f"possible_inside_relations:\n{possible_inside_relations}")
    # print(f"possible_outside_relations:\n{possible_outside_relations}")

    # # ratio
    # in_family_connection_statistics["empiric connection ratio 2 (mean ratio per family)"] = relations_statistics["ratio inner to outer relations"].mean()
    # probability
    in_family_connection_statistics["empiric connection probability 2 (mean probability per family)(higher better)"] = relations_statistics["inner connections probability"].mean()

    p_in_1 = relations_statistics["actual inside relations"].sum()/relations_statistics["possible inside relations"].sum()
    p_in_2 = relations_statistics["percentage inside relations"].mean()
    in_family_connection_statistics["percentage of possible in-family connections exists 1 (total/total)"] = p_in_1
    in_family_connection_statistics["percentage of possible in-family connections exists 2 (mean per family)"] = p_in_2

    # p_in = (not_mutually_related["same family"].sum() + mutually_related["same family"].sum()) / possible_inside_relations.sum()
    # print(f"{p_in:.2%} of possible in-family connections exists")
    # in_family_connection_statistics["percentage of possible in-family connections exists"] = p_in
    
    # out of family connection percentages
    p_out_1 = relations_statistics["actual outside relations"].sum()/relations_statistics["possible outside relations"].sum()
    p_out_2 = relations_statistics["percentage outside relations"].mean()
    in_family_connection_statistics["percentage of possible out of family connections exist 1 (total/total)"] = p_out_1
    in_family_connection_statistics["percentage of possible out of family connections exists 2 (mean per family)"] = p_out_2

    # p_out = ((len(not_mutually_related) - not_mutually_related["same family"].sum()) + (len(mutually_related) - mutually_related["same family"].sum())) / possible_outside_relations.sum()
    # print(f"{p_out:.2%} of possible out of family connections exists")
    # in_family_connection_statistics["percentage of possible out of family connections exists"] = p_out

    # # statistical connection ratios
    # in_family_connection_statistics["statistical connection ratio 1 (ratio of total/total)"] = p_in_1 / p_out_1
    # in_family_connection_statistics["statistical connection ratio 2 (ratio of mean per family)"] = p_in_2 / p_out_2
    # in_family_connection_statistics["statistical connection ratio 3 (mean of the ratios per family)"] = (relations_statistics["percentage inside relations"] / relations_statistics["percentage outside relations"]).mean()
    # # in_family_connection_statistics["statistical connection ratio"] = p_in / p_out
    # statistical connection probabilities
    in_family_connection_statistics["statistical connection probability 1 (Over-representation of total inner connections)(higher better)"] = (p_in_1 - p_out_1) / (p_in_1 + p_out_1)
    in_family_connection_statistics["statistical connection probability 2 (Over-representation of mean inner connections)(higher better)"] = (p_in_2 - p_out_2) / (p_in_2 + p_out_2)
    in_family_connection_statistics["statistical connection probability 3 (mean of Over-representation of inner connections per family)(higher better)"] = ((relations_statistics["percentage inside relations"] - relations_statistics["percentage outside relations"]) / (relations_statistics["percentage inside relations"] + relations_statistics["percentage outside relations"])).mean()

    # if a relations_graph was given, add the graph clustering indicators
    if pd.notna(relations_graph):
        regulations_DataFrame_F = regulations_DataFrame.copy().where(regulations_DataFrame["Control Identifier"].isin(relations_graph.nodes)).dropna()
        
        all_dists = dict(nx.all_pairs_dijkstra_path_length(relations_graph,weight='Weight'))
        # all_dists_df = pd.DataFrame(all_dists).T
        all_dists_df = pd.DataFrame.from_dict(all_dists)

        num_members_in_family = regulations_DataFrame_F.groupby(by=[family_column]).apply(len,include_groups=False)
        num_members_in_family.sum()
        node_family = regulations_DataFrame_F.set_index("Control Identifier", inplace=False)[family_column]
        # print(f"node_family = {node_family}")

        mean_node_to_family = pd.DataFrame(index=list(relations_graph.nodes),columns=regulations_DataFrame_F[family_column].unique(),dtype=float)
        mean_node_to_family
        mean_family_to_family = pd.DataFrame(index=regulations_DataFrame_F[family_column].unique(),columns=regulations_DataFrame_F[family_column].unique(),dtype=float)
        mean_family_to_family
        mean_node_to_all = all_dists_df.mean(axis="columns")
        mean_node_to_all
        mean_family_to_all = pd.Series(index=regulations_DataFrame_F[family_column].unique(),dtype=float)
        mean_family_to_all

        median_node_to_family = pd.DataFrame(index=list(relations_graph.nodes),columns=regulations_DataFrame_F[family_column].unique(),dtype=float)
        median_node_to_family
        median_family_to_family = pd.DataFrame(index=regulations_DataFrame_F[family_column].unique(),columns=regulations_DataFrame_F[family_column].unique(),dtype=float)
        median_family_to_family
        median_node_to_all = all_dists_df.median(axis="columns")
        median_node_to_all
        median_family_to_all = pd.Series(index=regulations_DataFrame_F[family_column].unique(),dtype=float)
        median_family_to_all

        for family1 in regulations_DataFrame_F[family_column].unique():
            # print(f"family1 is {family1}")
            family1_members = regulations_DataFrame_F[name_column].loc[regulations_DataFrame_F[family_column]==family1]
            num_members_in_family[family1] = len(family1_members)
            mean_family_to_all[family1] = all_dists_df.loc[:,family1_members].mean(axis=None)
            median_family_to_all[family1] = all_dists_df.loc[:,family1_members].median(axis=None)
            mean_node_to_family.loc[:,family1] = all_dists_df.loc[:,family1_members].mean(axis="columns")
            median_node_to_family.loc[:,family1] = all_dists_df.loc[:,family1_members].median(axis="columns")
            for family2 in regulations_DataFrame_F[family_column].unique():
                # print(f"family1 is {family2}")
                family2_members = regulations_DataFrame_F[name_column].loc[regulations_DataFrame_F[family_column]==family2]
                mean_family_to_family.loc[family1,family2] = all_dists_df.loc[family2_members,family1_members].mean(axis=None)
                median_family_to_family.loc[family1,family2] = all_dists_df.loc[family2_members,family1_members].median(axis=None)
                
        mean_family_to_all
        relations_statistics.loc[:,"mean dist from family to all"] = mean_family_to_all
        relations_statistics.loc[:,"median dist from family to all"] = median_family_to_all
        mean_node_to_family
        mean_family_to_family
        median_family_to_family

        # make final silhouette graph scores
        silhouette_score_g = pd.DataFrame(index=list(relations_graph.nodes),columns=["same_mean","same_median","closest_mean","closest_median"],dtype=float)
        for inx in mean_node_to_family.index:
            # print(f"inx = {inx}")
            # print(f"node_family[inx] = {node_family[inx]}")
            # print(f"mean_node_to_family.loc[inx,:] = {mean_node_to_family.loc[inx,:]}")
            # print(f"mean_node_to_family.loc[inx,node_family[inx]] = {mean_node_to_family.loc[inx,node_family[inx]]}")
            silhouette_score_g.loc[inx,"same_mean"] = mean_node_to_family.loc[inx,node_family[inx]]
            silhouette_score_g.loc[inx,"closest_mean"] = mean_node_to_family.loc[inx,:].drop(node_family[inx]).min()
            silhouette_score_g.loc[inx,"same_median"] = median_node_to_family.loc[inx,node_family[inx]]
            silhouette_score_g.loc[inx,"closest_median"] = median_node_to_family.loc[inx,:].drop(node_family[inx]).min()
        
        silhouette_score_g["silhouette_score_mean"] = (silhouette_score_g["closest_mean"] - silhouette_score_g["same_mean"]) / silhouette_score_g.loc[:,["closest_mean", "same_mean"]].max(axis="columns")
        silhouette_score_g["silhouette_score_median"] = (silhouette_score_g["closest_median"] - silhouette_score_g["same_median"]) / silhouette_score_g.loc[:,["closest_median", "same_median"]].max(axis="columns")
        silhouette_score_g
        # print(f"graph silhouette score is: {silhouette_score_g}")
        in_family_connection_statistics["graph silhouette score mean(higher better)"] = silhouette_score_g["silhouette_score_mean"].mean()
        # print(f"graph silhouette score mean is: {silhouette_score_g['silhouette_score_mean'].mean()}")
        in_family_connection_statistics["graph silhouette score median(higher better)"] = silhouette_score_g["silhouette_score_median"].median()
        # print(f"graph silhouette score median is: {silhouette_score_g['silhouette_score_median'].median()}")

        # make final calinski harabasz graph score
        mean_calinski_harabasz_BCSS = mean_family_to_all.copy() * num_members_in_family.copy()
        median_calinski_harabasz_BCSS = median_family_to_all.copy() * num_members_in_family.copy()
        mean_calinski_harabasz_WCSS = pd.Series(dtype=float)
        median_calinski_harabasz_WCSS = pd.Series(dtype=float)
        for family1 in regulations_DataFrame_F[family_column].unique():
            mean_calinski_harabasz_WCSS[family1] = mean_family_to_family.loc[family1,family1] * num_members_in_family[family1]
            median_calinski_harabasz_WCSS[family1] = median_family_to_family.loc[family1,family1] * num_members_in_family[family1]
        mean_calinski_harabasz_score_g = (mean_calinski_harabasz_BCSS.sum()/(regulations_DataFrame_F[family_column].nunique()-1)) / (mean_calinski_harabasz_WCSS.sum()/(len(relations_graph.nodes)-regulations_DataFrame_F[family_column].nunique()))
        median_calinski_harabasz_score_g = (median_calinski_harabasz_BCSS.sum()/(regulations_DataFrame_F[family_column].nunique()-1)) / (median_calinski_harabasz_WCSS.sum()/(len(relations_graph.nodes)-regulations_DataFrame_F[family_column].nunique()))
        mean_calinski_harabasz_score_g
        in_family_connection_statistics["graph calinski harabasz score mean(higher better)"] = mean_calinski_harabasz_score_g
        # print(f"graph mean calinski harabasz score is: {mean_calinski_harabasz_score_g}")
        in_family_connection_statistics["graph calinski harabasz score median(higher better)"] = median_calinski_harabasz_score_g
        # print(f"graph median calinski harabasz score is: {median_calinski_harabasz_score_g}")

        # make final davies bouldin graph score
        mean_davies_bouldin_S = pd.Series(dtype=float)
        median_davies_bouldin_S = pd.Series(dtype=float)
        for family1 in regulations_DataFrame_F[family_column].unique():
            mean_davies_bouldin_S[family1] = mean_family_to_family.loc[family1,family1]
            median_davies_bouldin_S[family1] = median_family_to_family.loc[family1,family1]
        relations_statistics.loc[:,"mean dist in family"] = mean_davies_bouldin_S
        relations_statistics.loc[:,"median dist in family"] = median_davies_bouldin_S
        mean_davies_bouldin_M = mean_family_to_family.copy()
        median_davies_bouldin_M = median_family_to_family.copy()
        mean_davies_bouldin_R = pd.DataFrame(dtype=float)
        median_davies_bouldin_R = pd.DataFrame(dtype=float)
        for family1 in regulations_DataFrame_F[family_column].unique():
            for family2 in regulations_DataFrame_F[family_column].unique():
                if family1 != family2:
                    mean_davies_bouldin_R.loc[family1,family2] = (mean_davies_bouldin_S[family1] + mean_davies_bouldin_S[family2]) / mean_davies_bouldin_M.loc[family1,family2]
                    median_davies_bouldin_R.loc[family1,family2] = (median_davies_bouldin_S[family1] + median_davies_bouldin_S[family2]) / median_davies_bouldin_M.loc[family1,family2]
        mean_davies_bouldin_score_g = mean_davies_bouldin_R.max().mean()
        mean_davies_bouldin_score_g
        median_davies_bouldin_score_g = median_davies_bouldin_R.max().median()
        median_davies_bouldin_score_g
        in_family_connection_statistics["graph davies bouldin score mean(lower better)"] = mean_davies_bouldin_score_g
        # print(f"graph mean davies bouldin score is: {mean_davies_bouldin_score_g}")
        in_family_connection_statistics["graph davies bouldin score median(lower better)"] = median_davies_bouldin_score_g
        # print(f"graph median davies bouldin score is: {median_davies_bouldin_score_g}")

        # make final dunn graph score
        MIN_mean = np.inf
        MIN_median = np.inf
        for i in mean_davies_bouldin_M.index:
            for j in mean_davies_bouldin_M.columns:
                if i != j and mean_davies_bouldin_M.loc[i,j] < MIN_mean:
                    MIN_mean = mean_davies_bouldin_M.loc[i,j]
                if i != j and median_davies_bouldin_M.loc[i,j] < MIN_median:
                    MIN_median = median_davies_bouldin_M.loc[i,j]
        mean_dunn_score_g = MIN_mean / mean_davies_bouldin_S.max()
        median_dunn_score_g = MIN_median / median_davies_bouldin_S.max()
        in_family_connection_statistics["graph dunn score mean(higher better)"] = mean_dunn_score_g
        # print(f"graph mean dunn score is: {mean_dunn_score_g}")
        in_family_connection_statistics["graph dunn score median(higher better)"] = median_dunn_score_g
        # print(f"graph median dunn score is: {median_dunn_score_g}")

    return mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics

def make_network_graph_from_relations(regulations_data:pd.DataFrame, relations_column:str|list[str], 
                                      sub_control_to_control_weight:float=0.1, 
                                      is_directed:bool=False, is_Main_Controls_Only:bool=True, 
                                      is_Weighted:bool=True, do_Aggregate_sub_Controls:bool=True) -> nx.Graph|nx.DiGraph:
    """
    Create a network graph from NIS800-53 regulatory relations data.

    This function constructs a graph representation of regulatory controls and their relationships
    based on the provided data. It can create either a directed or undirected graph, with options
    for weighting edges and focusing on main controls or including sub-controls.

    Parameters:
    regulations_data (pd.DataFrame): DataFrame containing the regulatory data.
    relations_column (str | list[str]): Column name(s) in the DataFrame that contain relation information.
    sub_control_to_control_weight (float): Weight to assign to edges between sub-controls and their main controls. Default is 0.1.
    is_directed (bool): If True, creates a directed graph. If False, creates an undirected graph. Default is False.
    is_Main_Controls_Only (bool): If True, only main controls are used as nodes. If False, includes sub-controls. Default is True.
    is_Weighted (bool): If True, edges are weighted based on relation strength. If False, all edges have equal weight. Default is True.
    do_Aggregate_sub_Controls (bool): If True, aggregates sub-controls under their main control. Default is True.

    Returns:
    nx.Graph | nx.DiGraph: A NetworkX graph object representing the regulatory relations network.
    """
    regulations_data = regulations_data.copy()
    # make a blank graph
    if is_directed:
        Related_Controls_Graph = nx.DiGraph()
    else:
        Related_Controls_Graph = nx.Graph()

    # add the nodes from the DataFrame to the graph
    if is_Main_Controls_Only:
        Related_Controls_Graph.add_nodes_from(regulations_data["Main Control Name"].unique())
    else:
        Related_Controls_Graph.add_nodes_from(regulations_data["Control Identifier"].unique())
    print(f"number of nodes in graph: {len(Related_Controls_Graph.nodes)}\nnumber of edges in graph: {len(Related_Controls_Graph.edges)}")
    
    # Make edges from every sub-control to its corresponding control
    if sub_control_to_control_weight > 0:
        for node in Related_Controls_Graph.nodes:
            # print(node)
            flag = node.find("(")
            if flag != -1:
                # add data to the node
                Related_Controls_Graph.nodes[node]["Main control"] = node[:flag]
                Related_Controls_Graph.nodes[node]["family"] = node[0:2]

                # print(f"is sub-control of {node[:flag]}") 
                if nx.is_directed(Related_Controls_Graph):
                    if is_Weighted:
                        Related_Controls_Graph.add_edge(node,node[:flag], Relations=1, Weight=sub_control_to_control_weight)
                        Related_Controls_Graph.add_edge(node[:flag],node, Relations=1, Weight=sub_control_to_control_weight)
                    else:
                        Related_Controls_Graph.add_edge(node,node[:flag])
                        Related_Controls_Graph.add_edge(node[:flag],node)
                else:
                    if is_Weighted:
                        Related_Controls_Graph.add_edge(node,node[:flag], Relations=2, Weight=sub_control_to_control_weight)
                    else:
                        Related_Controls_Graph.add_edge(node,node[:flag])
            else:
                # add data to the node
                Related_Controls_Graph.nodes[node]["Main control"] = node
                Related_Controls_Graph.nodes[node]["family"] = node[0:2]
        print(f"number of nodes in graph: {len(Related_Controls_Graph.nodes)}\nnumber of edges in graph: {len(Related_Controls_Graph.edges)}")

    # Make edges from every sub-control and control to its Related Controls from the relations_column
    for inx,row in regulations_data.iterrows():
        # print(row)
        if is_Main_Controls_Only:
            Control = row["Main Control Name"] # if we incorporate the sub-controls with the main control
        else:
            Control = row["Control Identifier"]

        # print(row["Control Identifier"])
        # Related_Controls = row[relations_column].keys()

        # for Related_Control in Related_Controls:
        if type(relations_column) == str:
            for Related_Control, value in row[relations_column].items():
                if is_Main_Controls_Only and Related_Control.find('(') != -1:
                    Related_Control = Related_Control[:Related_Control.find('(')]
                if not Related_Controls_Graph.has_node(Related_Control):
                    print(f"can't find control {Related_Control}")
                if Related_Control != "None" and Related_Control != Control:
                    if not Related_Controls_Graph.has_node(Related_Control):
                        print(f"can't find Related_Control {Related_Control}")
                    else: 
                        if is_Weighted:
                            if Related_Controls_Graph.has_edge(Control,Related_Control):
                                # add a Relation to the Relations counter
                                Related_Controls_Graph[Control][Related_Control]["Relations"] += 1
                                # Related_Controls_Graph.add_edge(Control,Related_Control, Relations=Related_Controls_Graph[Control][Related_Control]['Relations']+1)
                                # add the Relation Weight to the Weight_Sum counter
                                Related_Controls_Graph[Control][Related_Control]["Weight_Sum"] += (1-value)
                                # update the Weight 
                                ## min between current weight and new value
                                # Related_Controls_Graph[Control][Related_Control]["Weight"] = min((1-value),Related_Controls_Graph[Control][Related_Control]["Weight"])
                                ## update the Weight as the Weight_Sum divided by the number of Relations squared (more Relations means better weight)
                                Related_Controls_Graph[Control][Related_Control]["Weight"] = Related_Controls_Graph[Control][Related_Control]["Weight_Sum"] / (Related_Controls_Graph[Control][Related_Control]["Relations"]**2)
                            else:
                                Related_Controls_Graph.add_edge(Control,Related_Control, Relations=1, Weight=(1-value), Weight_Sum=(1-value))
                        else:
                            Related_Controls_Graph.add_edge(Control,Related_Control)
        elif type(relations_column) == list:
            for relations_type in relations_column:

                for Related_Control, value in row[relations_type].items():
                    if is_Main_Controls_Only and Related_Control.find('(') != -1:
                        Related_Control = Related_Control[:Related_Control.find('(')]
                    if not Related_Controls_Graph.has_node(Related_Control):
                        print(f"can't find control {Related_Control}")
                    if Related_Control != "None" and Related_Control != Control:
                        if not Related_Controls_Graph.has_node(Related_Control):
                            print(f"can't find Related_Control {Related_Control}")
                        else: 
                            if is_Weighted:
                                if Related_Controls_Graph.has_edge(Control,Related_Control):
                                    # add a Relation to the Relations counter
                                    Related_Controls_Graph[Control][Related_Control]["Relations"] += 1
                                    # Related_Controls_Graph.add_edge(Control,Related_Control, Relations=Related_Controls_Graph[Control][Related_Control]['Relations']+1)
                                    # add the Relation Weight to the Weight_Sum counter
                                    Related_Controls_Graph[Control][Related_Control]["Weight_Sum"] += (1-value)
                                    # update the Weight 
                                    ## min between current weight and new value
                                    # Related_Controls_Graph[Control][Related_Control]["Weight"] = min((1-value),Related_Controls_Graph[Control][Related_Control]["Weight"])
                                    ## update the Weight as the Weight_Sum divided by the number of Relations squared (more Relations means better weight)
                                    Related_Controls_Graph[Control][Related_Control]["Weight"] = Related_Controls_Graph[Control][Related_Control]["Weight_Sum"] / (Related_Controls_Graph[Control][Related_Control]["Relations"]**2)
                                else:
                                    Related_Controls_Graph.add_edge(Control,Related_Control, Relations=1, Weight=(1-value), Weight_Sum=(1-value))
                            else:
                                Related_Controls_Graph.add_edge(Control,Related_Control)

        # break
    print(f"number of nodes in graph: {len(Related_Controls_Graph.nodes)}\nnumber of edges in graph: {len(Related_Controls_Graph.edges)}")


    return Related_Controls_Graph

def inspect_connected_components(Related_Controls_Graph:nx.Graph|nx.DiGraph, 
                                 draw_kamada_kawai:bool=False, draw_spring:bool=False)-> tuple[list,list]|list:
    """
        Inspect and visualize connected components of a graph.

        This function analyzes the connected components of a given graph, prints information about them,
        and optionally visualizes the largest components using Kamada-Kawai and/or spring layouts.

        Parameters:
        Related_Controls_Graph (nx.Graph | nx.DiGraph): The input graph to be analyzed.
        draw_kamada_kawai (bool): If True, draw the largest component using Kamada-Kawai layout. Default is False.
        draw_spring (bool): If True, draw the largest component using spring layout. Default is False.

        Returns:
        tuple[list, list] | list: For directed graphs, returns a tuple containing lists of strongly and weakly
                                  connected components. For undirected graphs, returns a list of connected components.
                                  Components are sorted by size in descending order.
        """
    
    if Related_Controls_Graph.is_directed():
        # get the graph connected components
        strongly_connected_components = list(nx.strongly_connected_components(Related_Controls_Graph))
        strongly_connected_components = sorted(strongly_connected_components, key=lambda x: len(x), reverse=True)
        weakly_connected_components = list(nx.weakly_connected_components(Related_Controls_Graph))
        weakly_connected_components = sorted(weakly_connected_components, key=lambda x: len(x), reverse=True)

        # print the contents of the strongly connected components
        print(f"number of strongly connected components: {len(strongly_connected_components)}")

        for i,connected_component in enumerate(strongly_connected_components):
            print(f"component {i} has {len(connected_component)} nodes")
            if len(connected_component) <= 50:
                print(connected_component)

        # print the contents of the weakly connected components
        print(f"number of weakly connected components: {len(weakly_connected_components)}")

        for i,connected_component in enumerate(weakly_connected_components):
            print(f"component {i} has {len(connected_component)} nodes")
            if len(connected_component) <= 50:
                print(connected_component)

        # draw the biggest strongly connected component
        # print(f"largest strongly connected component of {len(strongly_connected_components[0])} nodes")
        if draw_kamada_kawai:
            Figure = plt.figure(figsize=(20,20))
            plt.title(f"Kamada-Kawai layout for largest strongly connected component of {len(strongly_connected_components[0])} nodes")
            nx.draw_kamada_kawai(Related_Controls_Graph.subgraph(strongly_connected_components[0]), with_labels=True, font_weight='bold')
        
        if draw_spring:
            Figure = plt.figure(figsize=(20,20))
            plt.title(f"Spring layout for largest strongly connected component of {len(strongly_connected_components[0])} nodes")
            nx.draw_spring(Related_Controls_Graph.subgraph(strongly_connected_components[0]), with_labels=True, font_weight='bold', edge_color='purple')
        
        # draw the biggest weakly connected component
        # print(f"largest weakly connected component of {len(weakly_connected_components[0])} nodes")
        
        if draw_kamada_kawai:
            Figure = plt.figure(figsize=(20,20))
            plt.title(f"Kamada-Kawai layout for largest weakly connected component of {len(weakly_connected_components[0])} nodes")
            nx.draw_kamada_kawai(Related_Controls_Graph.subgraph(weakly_connected_components[0]), with_labels=True, font_weight='bold')
        
        if draw_spring:
            Figure = plt.figure(figsize=(20,20))
            plt.title(f"Spring layout for largest weakly connected component of {len(weakly_connected_components[0])} nodes")
            nx.draw_spring(Related_Controls_Graph.subgraph(weakly_connected_components[0]), with_labels=True, font_weight='bold', edge_color='purple')
        
        
        return strongly_connected_components, weakly_connected_components
    else:
        # get the graph connected components
        connected_components = list(nx.connected_components(Related_Controls_Graph))
        connected_components = sorted(connected_components, key=lambda x: len(x), reverse=True)
        len(connected_components)

        # print the contents of the connected components
        print(f"number of connected components: {len(connected_components)}")

        for i,connected_component in enumerate(connected_components):
            print(f"component {i} has {len(connected_component)} nodes")
            if len(connected_component) <= 50:
                print(connected_component)

        # draw the biggest connected component
        # print(f"largest connected component of {len(connected_components[0])} nodes")
        if draw_kamada_kawai:
            Figure = plt.figure(figsize=(20,20))
            plt.title(f"Kamada-Kawai layout for largest connected component of {len(connected_components[0])} nodes")
            nx.draw_kamada_kawai(Related_Controls_Graph.subgraph(connected_components[0]), with_labels=True, font_weight='bold', )

        if draw_spring:
            Figure = plt.figure(figsize=(20,20))
            plt.title(f"Spring layout for largest connected component of {len(connected_components[0])} nodes")
            nx.draw_spring(Related_Controls_Graph.subgraph(connected_components[0]), with_labels=True, font_weight='bold', edge_color='purple')

        return connected_components

def cluster_analysis(Related_Controls_Graph:nx.Graph|nx.DiGraph, regulations_DataFrame:pd.DataFrame, 
                     relations_column:str|list[str], Is_Weighted:bool=True, Only_Main_Controls: bool=True, add_groups_summaries: bool=True,
                     do_Modularity_based_communities: bool=True, do_Louvain_communities: bool=True, do_Fluid_communities: bool=True,
                     do_Divisive_communities: bool=False, do_Label_propagation_communities: bool=False, do_Centrality_communities: bool=False,
                     )->tuple[pd.DataFrame,list[pd.DataFrame],pd.DataFrame]:
    """
    Perform cluster analysis on a graph of related controls using various community detection algorithms.

    Parameters:
    - Related_Controls_Graph (nx.Graph | nx.DiGraph): The graph representing related controls.
    - regulations_DataFrame (pd.DataFrame): DataFrame containing regulation data.
    - relations_column (str | list[str]): Column(s) in the DataFrame representing relationships.
    - Is_Weighted (bool): Whether the graph edges are weighted. Default is True.
    - Only_Main_Controls (bool): Whether the graph is full controls level or sub-controls level. Default is True(full controls level).
    - add_groups_summaries (bool): Whether to add group summaries to the results. Default is True.
    - do_Modularity_based_communities (bool): Whether to perform modularity-based community detection. Default is True.
    - do_Louvain_communities (bool): Whether to perform Louvain community detection. Default is True.
    - do_Fluid_communities (bool): Whether to perform fluid community detection. Default is True.
    - do_Divisive_communities (bool): Whether to perform divisive community detection. Default is False.
    - do_Label_propagation_communities (bool): Whether to perform label propagation community detection. Default is False.
    - do_Centrality_communities (bool): Whether to perform centrality-based community detection. Default is False.

    Returns: in_family_connection_statistics_test, relations_DataFrames, regulations_DataFrame
    tuple: A tuple containing three elements:
        - in_family_connection_statistics_test (pd.DataFrame): DataFrame of the statistics indicators compering all of the clustering methods.
        - relations_DataFrames (list[pd.DataFrame]): A list of DataFrames containing relations statistics for each clustering method.
        - regulations_DataFrame (pd.DataFrame): An updated regulations DataFrame with the results of the clustering methods.
    """

    if Only_Main_Controls:
        regulation_name_column:str = "Main Control Name"
    else:
        regulation_name_column:str = "Control Identifier"
    # base clusters
    print("-"*100)
    print("base NIS800-53 families:")
    mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame,relations_column,relations_graph=Related_Controls_Graph,name_column=regulation_name_column,is_Main_Controls_Only=Only_Main_Controls,do_print=False)
    relations_statistics.index.name = "original NIS families"
    in_family_connection_statistics_test = pd.DataFrame(index=in_family_connection_statistics.index)
    in_family_connection_statistics_test.loc[:,"NIS800-53 families"] = in_family_connection_statistics
    relations_DataFrames = []

    if add_groups_summaries:
        relations_statistics = add_summarization(regulations_DataFrame, relations_statistics.copy(),is_Main_Controls_Only=Only_Main_Controls)

    relations_DataFrames.append(relations_statistics)
    print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection probability 1 (same family/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection probability 2 (mean probability per family)(higher better)']:.3}")
    print(in_family_connection_statistics)

    # Modularity-based communities
    if do_Modularity_based_communities:
        print("-"*100)
        print("Modularity-based communities:")

        if Is_Weighted:
            Modularity_based_communities = nx.community.greedy_modularity_communities(Related_Controls_Graph, weight='Weight')
        else:
            Modularity_based_communities = nx.community.greedy_modularity_communities(Related_Controls_Graph)


        regulations_DataFrame = print_and_write_communities_results(Modularity_based_communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Modularity_based_communities",node_column = regulation_name_column)
        mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = regulation_name_column, family_column = "Modularity_based_communities",is_Main_Controls_Only=Only_Main_Controls, do_print = False)
        # if Only_Main_Controls:    
        #     regulations_DataFrame = print_and_write_communities_results(Modularity_based_communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Modularity_based_communities",node_column = "Main Control Name")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Main Control Name", family_column = "Modularity_based_communities", do_print = False)
        # else:
        #     regulations_DataFrame = print_and_write_communities_results(Modularity_based_communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Modularity_based_communities",node_column = "Control Identifier")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Control Identifier", family_column = "Modularity_based_communities", do_print = False)
            
        # print_communities_results(Modularity_based_communities,regulations_DataFrame,max_nodes_to_print=30)

        # print(f"there are {len(not_mutually_related)} non-mutual connections")

        # print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection ratio 1 (total/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection ratio 2 (mean ratio per family)(higher better)']:.3}")
        print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection probability 1 (same family/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection probability 2 (mean probability per family)(higher better)']:.3}")

        # not_mutually_related
        Modularity_based_communities_relations_statistics = relations_statistics # prepare for output file
        
        if add_groups_summaries:
            relations_statistics = add_summarization(regulations_DataFrame, relations_statistics.copy(),is_Main_Controls_Only=Only_Main_Controls)
        
        relations_DataFrames.append(relations_statistics)
        print(in_family_connection_statistics)
        in_family_connection_statistics_test.loc[:,"Modularity based communities"] = in_family_connection_statistics


    # Louvain Community Detection
    if do_Louvain_communities:
        print("-"*100)
        print("Louvain Community Detection:")

        if Is_Weighted:
            Louvain_Communities = nx.community.louvain_communities(Related_Controls_Graph, weight='Weight')
        else:
            Louvain_Communities = nx.community.louvain_communities(Related_Controls_Graph)


        regulations_DataFrame = print_and_write_communities_results(Louvain_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Louvain_Communities",node_column = regulation_name_column)
        mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = regulation_name_column, family_column = "Louvain_Communities",is_Main_Controls_Only=Only_Main_Controls, do_print = False)
        # if Only_Main_Controls:    
        #     regulations_DataFrame = print_and_write_communities_results(Louvain_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Louvain_Communities",node_column = "Main Control Name")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Main Control Name", family_column = "Louvain_Communities", do_print = False)
        # else:
        #     regulations_DataFrame = print_and_write_communities_results(Louvain_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Louvain_Communities",node_column = "Control Identifier")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Control Identifier", family_column = "Louvain_Communities", do_print = False)

        # print_communities_results(Louvain_Communities,regulations_DataFrame,max_nodes_to_print=30)

        # print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection ratio 1 (total/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection ratio 2 (mean ratio per family)(higher better)']:.3}")
        print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection probability 1 (same family/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection probability 2 (mean probability per family)(higher better)']:.3}")

        Louvain_Communities_relations_statistics = relations_statistics # prepare for output file

        if add_groups_summaries:
            relations_statistics = add_summarization(regulations_DataFrame, relations_statistics.copy(),is_Main_Controls_Only=Only_Main_Controls)

        relations_DataFrames.append(relations_statistics)
        print(in_family_connection_statistics)
        in_family_connection_statistics_test.loc[:,"Louvain Communities"] = in_family_connection_statistics


    # Fluid Communities
    if do_Fluid_communities:
        print("-"*100)
        print("Fluid Communities:")
        
        k = regulations_DataFrame["Family Name"].nunique()
        
        # Fluid Communities can only take undirected graphs
        if nx.is_directed(Related_Controls_Graph):
            Fluid_Communities = nx.community.asyn_fluidc(nx.to_undirected(Related_Controls_Graph), k, max_iter=10_000)
        else:
            Fluid_Communities = nx.community.asyn_fluidc(Related_Controls_Graph, k, max_iter=10_000)


        # print_communities_results(Fluid_Communities,regulations_DataFrame,max_nodes_to_print=30,k=k)
        regulations_DataFrame = print_and_write_communities_results(Fluid_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Fluid_Communities",node_column = regulation_name_column)
        mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = regulation_name_column, family_column = "Fluid_Communities",is_Main_Controls_Only=Only_Main_Controls, do_print = False)
        # if Only_Main_Controls:    
        #     regulations_DataFrame = print_and_write_communities_results(Fluid_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,k=k,communities_name="Fluid_Communities",node_column = "Main Control Name")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Main Control Name", family_column = "Fluid_Communities", do_print = False)
        # else:
        #     regulations_DataFrame = print_and_write_communities_results(Fluid_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,k=k,communities_name="Fluid_Communities",node_column = "Control Identifier")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Control Identifier", family_column = "Fluid_Communities", do_print = False)


        # print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection ratio 1 (total/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection ratio 2 (mean ratio per family)(higher better)']:.3}")
        print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection probability 1 (same family/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection probability 2 (mean probability per family)(higher better)']:.3}")

        Fluid_Communities_relations_statistics = relations_statistics # prepare for output file
        
        if add_groups_summaries:
            relations_statistics = add_summarization(regulations_DataFrame, relations_statistics.copy(),is_Main_Controls_Only=Only_Main_Controls)
        
        relations_DataFrames.append(relations_statistics)
        print(in_family_connection_statistics)
        in_family_connection_statistics_test.loc[:,"Fluid Communities"] = in_family_connection_statistics

    # Divisive Communities
    if do_Divisive_communities: 
        print("-"*100)
        print("Divisive Communities:")
        
        k = regulations_DataFrame["Family Name"].nunique()
        if nx.is_directed(Related_Controls_Graph):
            if Is_Weighted:
                Divisive_Communities = nx.community.edge_betweenness_partition(nx.to_undirected(Related_Controls_Graph),k, weight='Weight')
                # Divisive_Communities = nx.community.edge_current_flow_betweenness_partition(nx.to_undirected(Related_Controls_Graph),k, weight='Weight')
            else:
                Divisive_Communities = nx.community.edge_betweenness_partition(nx.to_undirected(Related_Controls_Graph),k)
                # Divisive_Communities = nx.community.edge_current_flow_betweenness_partition(nx.to_undirected(Related_Controls_Graph),k)
        else:
            if Is_Weighted:
                Divisive_Communities = nx.community.edge_betweenness_partition(Related_Controls_Graph,k, weight='Weight')
                # Divisive_Communities = nx.community.edge_current_flow_betweenness_partition(Related_Controls_Graph,k, weight='Weight')
            else:
                Divisive_Communities = nx.community.edge_betweenness_partition(Related_Controls_Graph,k)
                # Divisive_Communities = nx.community.edge_current_flow_betweenness_partition(Related_Controls_Graph,k)

        
        regulations_DataFrame = print_and_write_communities_results(Divisive_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Divisive_Communities",node_column = regulation_name_column)
        mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = regulation_name_column, family_column = "Divisive_Communities",is_Main_Controls_Only=Only_Main_Controls, do_print = False)
        # if Only_Main_Controls:    
        #     regulations_DataFrame = print_and_write_communities_results(Divisive_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,k=k,communities_name="Divisive Communities",node_column = "Main Control Name")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Main Control Name", family_column = "Divisive Communities", do_print = False)
        # else:
        #     regulations_DataFrame = print_and_write_communities_results(Divisive_Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,k=k,communities_name="Divisive Communities",node_column = "Control Identifier")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Control Identifier", family_column = "Divisive Communities", do_print = False)


        # print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection ratio 1 (total/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection ratio 2 (mean ratio per family)(higher better)']:.3}")
        print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection probability 1 (same family/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection probability 2 (mean probability per family)(higher better)']:.3}")

        Divisive_Communities_relations_statistics = relations_statistics # prepare for output file
        
        if add_groups_summaries:
            relations_statistics = add_summarization(regulations_DataFrame, relations_statistics.copy(),is_Main_Controls_Only=Only_Main_Controls)
        
        relations_DataFrames.append(relations_statistics)
        print(in_family_connection_statistics)
        in_family_connection_statistics_test.loc[:,"Divisive Communities"] = in_family_connection_statistics

    # Label propagation communities
    if do_Label_propagation_communities:  
        print("-"*100)
        print("Label Propagation communities:")
        
        if Is_Weighted:
            Label_propagation = nx.community.asyn_lpa_communities(Related_Controls_Graph, weight='Weight')
            # Label_propagation = nx.community.label_propagation_communities(Related_Controls_Graph)
            # Label_propagation = nx.community.fast_label_propagation_communities(Related_Controls_Graph, weight='Weight')
        else:
            Label_propagation = nx.community.asyn_lpa_communities(Related_Controls_Graph)
            # Label_propagation = nx.community.label_propagation_communities(Related_Controls_Graph)
            # Label_propagation = nx.community.fast_label_propagation_communities(Related_Controls_Graph)

        regulations_DataFrame = print_and_write_communities_results(Label_propagation,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Label_propagation",node_column = regulation_name_column)
        mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = regulation_name_column, family_column = "Label_propagation",is_Main_Controls_Only=Only_Main_Controls, do_print = False)
        # if Only_Main_Controls:    
        #     regulations_DataFrame = print_and_write_communities_results(Label_propagation,regulations_DataFrame.copy(),max_nodes_to_print=30,k=k,communities_name="Label propagation",node_column = "Main Control Name")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Main Control Name", family_column = "Label propagation", do_print = False)
        # else:
        #     regulations_DataFrame = print_and_write_communities_results(Label_propagation,regulations_DataFrame.copy(),max_nodes_to_print=30,k=k,communities_name="Label propagation",node_column = "Control Identifier")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Control Identifier", family_column = "Label propagation", do_print = False)


        # print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection ratio 1 (total/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection ratio 2 (mean ratio per family)(higher better)']:.3}")
        print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection probability 1 (same family/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection probability 2 (mean probability per family)(higher better)']:.3}")

        Label_propagation_relations_statistics = relations_statistics # prepare for output file
        
        if add_groups_summaries:
            relations_statistics = add_summarization(regulations_DataFrame, relations_statistics.copy(),is_Main_Controls_Only=Only_Main_Controls)
        
        relations_DataFrames.append(relations_statistics)
        print(in_family_connection_statistics)
        in_family_connection_statistics_test.loc[:,"Label propagation"] = in_family_connection_statistics

    # Partitions via centrality measures
    if do_Centrality_communities:  
        print("-"*100)
        print("Partitions via Centrality Measures:")
        
        centrality_Communities = nx.community.girvan_newman(Related_Controls_Graph)

        k = regulations_DataFrame["Family Name"].nunique()
        for Communities in centrality_Communities:
            print(len(Communities))
            if len(Communities) < k:
                print(Communities)
            elif len(Communities) == k:
                print(Communities)
                break
        
        regulations_DataFrame = print_and_write_communities_results(Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,communities_name="Centrality_Communities",node_column = regulation_name_column)
        mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = regulation_name_column, family_column = "Centrality_Communities",is_Main_Controls_Only=Only_Main_Controls, do_print = False)
        # if Only_Main_Controls:    
        #     regulations_DataFrame = print_and_write_communities_results(Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,k=k,communities_name="Centrality_Communities",node_column = "Main Control Name")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Main Control Name", family_column = "Centrality Communities", do_print = False)
        # else:
        #     regulations_DataFrame = print_and_write_communities_results(Communities,regulations_DataFrame.copy(),max_nodes_to_print=30,k=k,communities_name="Centrality Communities",node_column = "Control Identifier")
        #     mutually_related, not_mutually_related, in_family_connection_statistics, relations_statistics = get_mutual_relations(regulations_DataFrame, relations_column, relations_graph=Related_Controls_Graph, name_column = "Control Identifier", family_column = "Centrality Communities", do_print = False)


        # print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection ratio 1 (total/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection ratio 2 (mean ratio per family)(higher better)']:.3}")
        print(f"the in_family_connection_ratios are: {in_family_connection_statistics['empiric connection probability 1 (same family/total)(higher better)']:.3} , {in_family_connection_statistics['empiric connection probability 2 (mean probability per family)(higher better)']:.3}")

        Centrality_Communities_relations_statistics = relations_statistics # prepare for output file
        
        if add_groups_summaries:
            relations_statistics = add_summarization(regulations_DataFrame, relations_statistics.copy(),is_Main_Controls_Only=Only_Main_Controls)
        
        relations_DataFrames.append(relations_statistics)
        print(in_family_connection_statistics)
        in_family_connection_statistics_test.loc[:,"Centrality Communities"] = in_family_connection_statistics


    return in_family_connection_statistics_test, relations_DataFrames, regulations_DataFrame




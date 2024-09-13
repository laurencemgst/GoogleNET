import pandas as pd
import streamlit as st
from datasets import load_dataset
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
ds = load_dataset("lllaurenceee/Shopee_Bicycle_Reviews")
df = ds['train'].to_pandas()

# Group by 'orderid' and aggregate 'purchased_item' lists
grouped = df.groupby('orderid')['purchased_item'].apply(list).reset_index()

# Filter out transactions with only one item
filtered = grouped[grouped['purchased_item'].apply(len) > 1]

# Convert the 'purchased_item' column to a list of transactions
transactions = filtered['purchased_item'].tolist()

# Encode the transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)

# Convert itemsets to frozensets for better readability
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: frozenset(x))

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Convert itemsets, antecedents, and consequents to string for better readability
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Print results
st.write("Frequent Itemsets:")
st.write(frequent_itemsets)

st.write("\nAssociation Rules:")
st.write(rules)

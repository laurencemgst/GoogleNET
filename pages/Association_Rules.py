import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title('Association Rules Visualization')

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("hf://datasets/lllaurenceee/Shopee_Bicycle_Reviews/Dataset_D_Duplicate.csv")

df = load_data()

# Data Preparation
grouped = df.groupby('orderid')['purchased_item'].apply(list).reset_index()
filtered = grouped[grouped['purchased_item'].apply(len) > 1]
transactions = filtered['purchased_item'].tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: frozenset(x))
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# Plot Functions
def plot_bar(data, x_col, y_col, title, xlabel, ylabel, color='skyblue'):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=data, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(plt.gcf())

# Frequent Itemsets
st.markdown("### Top 10 Frequent Itemsets")
plot_bar(frequent_itemsets.head(10), 'support', 'itemsets', 'Top 10 Frequent Itemsets', 'Support', 'Itemsets')

# Top 10 Most Purchased Items
st.markdown("### Top 10 Most Purchased Items")
item_counts = df['purchased_item'].value_counts().head(10)
plot_bar(pd.DataFrame({'itemsets': item_counts.index, 'support': item_counts.values}), 'support', 'itemsets', 'Top 10 Most Purchased Items', 'Number of Purchases', 'Purchased Item')

# Top 10 Shops by Sales Volume
st.markdown("### Top 10 Shops by Sales Volume")
shop_sales = df['shop'].value_counts().head(10)
plot_bar(pd.DataFrame({'shop': shop_sales.index, 'sales': shop_sales.values}), 'sales', 'shop', 'Top 10 Shops by Sales Volume', 'Number of Orders', 'Shop')

# Top 10 Products by Average Rating
st.markdown("### Top 10 Products by Average Rating")
avg_rating = df.groupby('purchased_item')['rating'].mean().sort_values(ascending=False).head(10)
plot_bar(pd.DataFrame({'product': avg_rating.index, 'rating': avg_rating.values}), 'rating', 'product', 'Top 10 Products by Average Rating', 'Average Rating', 'Product Name')

# Price Distribution
st.markdown("### Price Distribution of Products")
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], bins=20, kde=True, color='blue')
plt.title('Price Distribution of Products', fontsize=14)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
st.pyplot(plt.gcf())

# Top 10 Product Variations
st.markdown("### Top 10 Product Variations")
variation_counts = df['variation'].value_counts().head(10)
plot_bar(pd.DataFrame({'variation': variation_counts.index, 'count': variation_counts.values}), 'count', 'variation', 'Top 10 Product Variations', 'Number of Purchases', 'Variation')

# Monthly Sales Trends
st.markdown("### Monthly Sales Trends")
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').astype(str)
monthly_sales = df['month'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o', color='teal')
plt.title('Monthly Sales Trends', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt.gcf())

# Top Items by Month
st.markdown("### Top 5 Items by Month")
df['month'] = df['date'].dt.to_period('M').astype(str)
basket = pd.pivot_table(df, index='orderid', columns='purchased_item', aggfunc='size', fill_value=0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

for month in df['month'].unique():
    monthly_data = basket.loc[df[df['month'] == month]['orderid'].unique()]
    frequent_items = apriori(monthly_data, min_support=0.01, use_colnames=True)
    top_items = frequent_items.sort_values(by='support', ascending=False).head(5)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_items['support'], y=top_items['itemsets'].apply(lambda x: ', '.join(list(x))), palette='Blues_d')
    plt.title(f'Top 5 Items with Highest Support - {month}', fontsize=16)
    plt.xlabel('Support', fontsize=14)
    plt.ylabel('Itemsets', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt.gcf())

# High Confidence Rules
st.markdown("### Itemsets with High Confidence")
confidence_threshold = 0.54
high_confidence_rules = rules[rules['confidence'] > confidence_threshold]
high_confidence_rules = high_confidence_rules.sort_values(by='confidence', ascending=False)
table = high_confidence_rules[['antecedents', 'consequents', 'support', 'confidence']]
st.write(table)

# High Support Rules (continued)
support_threshold = 0.05
high_support_rules = rules[rules['support'] > support_threshold]
high_support_rules = high_support_rules.sort_values(by='support', ascending=False)
table = high_support_rules[['antecedents', 'consequents', 'support']]
st.write(table)

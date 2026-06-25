import pandas as pd

orders = pd.read_csv("data/List of Orders.csv")
details = pd.read_csv("data/Order Details.csv")
target = pd.read_csv("data/Sales target.csv")

print("Orders Shape:", orders.shape)
print("Order Details Shape:", details.shape)
print("Sales Target Shape:", target.shape)
print("\nOrders Data")
print(orders.head())

print("\nOrder Details Data")
print(details.head())

print("\nSales Target Data")
print(target.head())
print("\n========== Orders Info ==========")
orders.info()

print("\n========== Order Details Info ==========")
details.info()

print("\n========== Sales Target Info ==========")
target.info()
print("\n========== Missing Values ==========")

print("\nOrders")
print(orders.isnull().sum())

print("\nOrder Details")
print(details.isnull().sum())

print("\nSales Target")
print(target.isnull().sum())
print("\n========== Duplicate Records ==========")

print("Orders :", orders.duplicated().sum())
print("Order Details :", details.duplicated().sum())
print("Sales Target :", target.duplicated().sum())
print("\n========== Numerical Summary ==========")
print(details.describe())
print("\nOrders Columns")
print(orders.columns)

print("\nOrder Details Columns")
print(details.columns)

print("\nSales Target Columns")
print(target.columns)
sales = pd.merge(
    orders,
    details,
    on="Order ID",
    how="inner"
)
print(sales.head())
print(sales.shape)
# ============================================
# Merged Dataset Information
# ============================================

print("\n========== Merged Dataset Info ==========")
sales.info()

print("\n========== Missing Values in Merged Dataset ==========")
print(sales.isnull().sum())

print("\n========== Duplicate Records in Merged Dataset ==========")
print(sales.duplicated().sum())
# ============================================
# Convert Order Date to DateTime
# ============================================

sales["Order Date"] = pd.to_datetime(
    sales["Order Date"],
    format="%d-%m-%Y"
)
# ============================================
# Feature Engineering
# ============================================

sales["Year"] = sales["Order Date"].dt.year
sales["Month"] = sales["Order Date"].dt.month_name()
sales["Quarter"] = sales["Order Date"].dt.quarter
sales["Day"] = sales["Order Date"].dt.day_name()

print(sales.head())
# ============================================
# Business KPIs
# ============================================

print("\n========== Business KPIs ==========")

print("Total Sales :", sales["Amount"].sum())
print("Total Profit :", sales["Profit"].sum())
print("Total Orders :", sales["Order ID"].nunique())
print("Total Quantity Sold :", sales["Quantity"].sum())
print("Average Sales :", round(sales["Amount"].mean(),2))
print("Average Profit :", round(sales["Profit"].mean(),2))
# ==========================================
# Top 10 States by Sales
# ==========================================

state_sales = sales.groupby("State")["Amount"].sum().sort_values(ascending=False)

print("\nTop States by Sales")
print(state_sales.head(10))
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

state_sales.head(10).plot(kind="bar")

plt.title("Top 10 States by Sales")
plt.xlabel("State")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("top_10_states_sales.png", dpi=300, bbox_inches="tight")
# plt.show()


# ==========================================
# Monthly Sales Trend
# ==========================================

monthly_sales = sales.groupby("Month")["Amount"].sum()

print("\nMonthly Sales")
print(monthly_sales)
plt.figure(figsize=(10,5))

monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid(True)

plt.tight_layout()
plt.savefig("monthly_sales_trend.png", dpi=300, bbox_inches="tight")
# plt.show()
# ==========================================
# Category Wise Sales
# ==========================================
category_sales = sales.groupby("Category")["Amount"].sum().sort_values(ascending=False)

print("\nCategory Wise Sales")
print(category_sales)

plt.figure(figsize=(8,5))

category_sales.plot(kind="bar")

plt.title("Category Wise Sales")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig("category_sales.png", dpi=300, bbox_inches="tight")

# plt.show()
# ==========================================
# Category Wise Profit
# ==========================================
category_profit = sales.groupby("Category")["Profit"].sum().sort_values(ascending=False)

print("\nCategory Wise Profit")
print(category_profit)

plt.figure(figsize=(8,5))

category_profit.plot(kind="bar")

plt.title("Category Wise Profit")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig("category_profit.png", dpi=300, bbox_inches="tight")

# plt.show()
# ==========================================
# State Wise Profit
# ==========================================
state_profit = sales.groupby("State")["Profit"].sum().sort_values(ascending=False).head(10)

print("\nTop 10 States by Profit")
print(state_profit)

plt.figure(figsize=(10,5))

state_profit.plot(kind="bar")

plt.title("Top 10 States by Profit")
plt.xlabel("State")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig("top_10_states_profit.png", dpi=300, bbox_inches="tight")

# plt.show()
# ==========================================
# Top 10 Customers by Sales
# ==========================================
customer_sales = (
    sales.groupby("CustomerName")["Amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Customers by Sales")
print(customer_sales)

plt.figure(figsize=(10,5))

customer_sales.plot(kind="bar")

plt.title("Top 10 Customers by Sales")
plt.xlabel("Customer")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("top_10_customers_sales.png", dpi=300, bbox_inches="tight")

# plt.show()
# ==========================================
# Top 10 Sub-Category by Sales
# ==========================================
subcategory_sales = (
    sales.groupby("Sub-Category")["Amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Sub-Categories by Sales")
print(subcategory_sales)

plt.figure(figsize=(10,5))

subcategory_sales.plot(kind="bar")

plt.title("Top 10 Sub-Categories by Sales")
plt.xlabel("Sub-Category")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("top_10_subcategory_sales.png", dpi=300, bbox_inches="tight")

# plt.show()
# ============================================
# Top 10 Customers by Profit
# ============================================

customer_profit = sales.groupby("CustomerName")["Profit"].sum().sort_values(ascending=False).head(10)

print("\nTop 10 Customers by Profit")
print(customer_profit)

plt.figure(figsize=(10,5))

customer_profit.plot(kind="bar")

plt.title("Top 10 Customers by Profit")
plt.xlabel("Customer")
plt.ylabel("Profit")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("top_10_customers_profit.png", dpi=300, bbox_inches="tight")

# plt.show()
# ============================================
# Monthly Profit Trend
# ============================================

monthly_profit = sales.groupby("Month")["Profit"].sum()

print("\nMonthly Profit")
print(monthly_profit)

plt.figure(figsize=(10,5))

monthly_profit.plot(kind="line", marker="o")

plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")

plt.grid(True)

plt.tight_layout()

plt.savefig("monthly_profit_trend.png", dpi=300, bbox_inches="tight")

# plt.show()
# ============================================
# Top 10 Loss Making Sub-Categories
# ============================================
loss_subcategory = (
    sales.groupby("Sub-Category")["Profit"]
    .sum()
    .sort_values()
    .head(10)
)

print("\nTop 10 Loss Making Sub-Categories")
print(loss_subcategory)

plt.figure(figsize=(10,5))

loss_subcategory.plot(kind="bar", color="red")

plt.title("Top 10 Loss Making Sub-Categories")
plt.xlabel("Sub-Category")
plt.ylabel("Profit")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("loss_subcategory_profit.png", dpi=300, bbox_inches="tight")

# plt.show()
# ============================================
# Top 10 Most Profitable Sub-Categories
# ============================================
profit_subcategory = (
    sales.groupby("Sub-Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Most Profitable Sub-Categories")
print(profit_subcategory)

plt.figure(figsize=(10,5))

profit_subcategory.plot(kind="bar")

plt.title("Top 10 Most Profitable Sub-Categories")
plt.xlabel("Sub-Category")
plt.ylabel("Profit")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("top_10_subcategory_profit.png", dpi=300, bbox_inches="tight")

# plt.show()
# ============================================
# Sales vs Profit Scatter Plot
# ============================================
plt.figure(figsize=(10,6))

plt.scatter(
    sales["Amount"],
    sales["Profit"],
    alpha=0.6
)

plt.title("Sales vs Profit")
plt.xlabel("Sales Amount")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig("sales_vs_profit_scatter.png", dpi=300, bbox_inches="tight")

# plt.show()
# ============================================
# Correlation Heatmap
# ============================================
import seaborn as sns

plt.figure(figsize=(8,6))

sns.heatmap(
    sales[["Amount","Profit","Quantity"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")

# plt.show()
# ============================================
# Category Profit Margin
# ============================================
category_summary = (
    sales.groupby("Category")[["Amount", "Profit"]]
    .sum()
)

category_summary["Profit Margin (%)"] = (
    category_summary["Profit"] / category_summary["Amount"]
) * 100

print("\nCategory Profit Margin")
print(category_summary.round(2))
# ============================================
# Top 5 Loss Making Customers
# ============================================
loss_customers = (
    sales.groupby("CustomerName")["Profit"]
    .sum()
    .sort_values()
    .head(5)
)

print("\nTop 5 Loss Making Customers")
print(loss_customers)
# ============================================
# State Wise Sales and Profit Summary
# ============================================
state_summary = (
    sales.groupby("State")[["Amount", "Profit", "Quantity"]]
    .sum()
    .sort_values(by="Amount", ascending=False)
)

print("\nState Wise Sales and Profit Summary")
print(state_summary)
# ============================================
# Top 5 Best Performing States
# ============================================
top_states = state_summary.head(5)

print("\nTop 5 Best Performing States")
print(top_states)
# ============================================
# Export Final Clean Dataset
# ============================================
sales.to_csv("sales_cleaned.csv", index=False)

print("\n✅ sales_cleaned.csv exported successfully.")


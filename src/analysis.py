import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load the e-commerce sales dataset."""
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    return df


def clean_data(df):
    """Clean and prepare the dataset."""
    df = df.copy()

    df.drop_duplicates(inplace=True)

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(
            df["Order Date"],
            errors="coerce",
            dayfirst=True
        )

    return df


def sales_by_category(df):
    """Calculate total sales by product category."""
    return (
        df.groupby("Product Category")["Price"]
        .sum()
        .sort_values(ascending=False)
    )


def top_products(df, n=10):
    """Return top products based on total sales."""
    return (
        df.groupby("Product Name")["Price"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )


def sales_by_city(df):
    """Calculate total sales by city."""
    return (
        df.groupby("City")["Price"]
        .sum()
        .sort_values(ascending=False)
    )


def sales_by_gender(df):
    """Calculate total sales by gender."""
    return (
        df.groupby("Gender")["Price"]
        .sum()
        .sort_values(ascending=False)
    )


def sales_by_payment_method(df):
    """Calculate total sales by payment method."""
    return (
        df.groupby("Payment Method")["Price"]
        .sum()
        .sort_values(ascending=False)
    )


def monthly_sales(df):
    """Calculate monthly sales trend."""
    data = df.dropna(subset=["Order Date"]).copy()
    data["Month"] = data["Order Date"].dt.to_period("M")

    return data.groupby("Month")["Price"].sum()


def average_order_value(df):
    """Calculate average order value."""
    return df["Price"].mean()


def plot_sales(data, title, xlabel, ylabel="Total Sales"):
    """Create a reusable bar chart."""
    data.plot(kind="bar", figsize=(10, 6))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    DATA_PATH = "data/ecommerce_sales_data.csv"

    df = load_data(DATA_PATH)
    df = clean_data(df)

    print("\nSales by Product Category:")
    print(sales_by_category(df))

    print("\nTop 10 Products:")
    print(top_products(df))

    print("\nSales by City:")
    print(sales_by_city(df))

    print("\nSales by Gender:")
    print(sales_by_gender(df))

    print("\nSales by Payment Method:")
    print(sales_by_payment_method(df))

    print("\nMonthly Sales:")
    print(monthly_sales(df))

    print("\nAverage Order Value:")
    print(f"{average_order_value(df):,.2f}")

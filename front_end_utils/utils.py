import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import base64
import io, sys

def show_heat_map():
    df = pd.read_csv('Notebook\Datasets\pre_processed_data.csv')
    df.drop(columns='class', inplace=True)
    cor = df.corr()
    fig = Figure(figsize=(10,10))
    ax= fig.add_subplot(1,1,1)
    sns.heatmap(cor, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode('ascii')
    buf.close()
    return f'data:image/png;base64,{encoded_img}'
def show_pie_chart():
    df = pd.read_csv('Notebook/Datasets/pre_processed_data.csv')

    class_counts = df['class'].value_counts()
    
    # Fix: Use plt.figure() instead of plt.subplots()
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['#A9B5DF', '#7886C7'])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode('ascii')
    buf.close()
    
    return f'data:image/png;base64,{encoded_img}'
def tracking_trends():
    df = pd.read_csv('Notebook/Datasets/pre_processed_data.csv')
    df.drop(columns='class', inplace=True)

    num_cols = len(df.columns)
    rows = (num_cols + 1) // 2  # Dynamic row calculation
    fig = Figure(figsize=(12, rows * 4))
    
    for idx, col in enumerate(df.columns, 1):
        ax = fig.add_subplot(rows, 2, idx)
        sns.histplot(df[col], kde=True, bins=20, color='#2D336B', ax=ax,
                     line_kws={'color': 'white'})  # Corrected KDE line color
        ax.set_title(f'Distribution of {col}', fontsize=12)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
    
    fig.subplots_adjust(hspace=0.4)  # Add space between subplots

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode('ascii')
    buf.close()
    
    return f'data:image/png;base64,{encoded_img}'


def prediction_pie(prob_bankrupt, prob_not_bankrupt):
    plt.figure(figsize=(6, 6))
    plt.pie(x=[prob_bankrupt, prob_not_bankrupt], labels=["Bankrupt Percentage", "Not Bankrupt Percentage"], autopct='%1.1f%%', colors=['#A9B5DF', '#7886C7'])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode('ascii')
    buf.close()
    
    return f'data:image/png;base64,{encoded_img}'

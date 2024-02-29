from plotnine import ggplot, aes, geom_point, labs, facet_wrap, geom_vline, theme_minimal, theme, element_text, scale_shape_manual
import pandas as pd


# PATHS (edit these paths depending on dataset)
figures_path = 'figures/genome'
acc_rate_csv_path = 'acc_rate/genome.csv'
dataset_name = 'genome'



def acc_plot(df):
    # Creating data for fold 1
    data_fold1 = pd.DataFrame({
        'Method': df['method'],
        'Accuracy': df['fold1.test'],
        'Fold': 'Fold 1 Test'
    })

    # Creating data for fold 2
    data_fold2 = pd.DataFrame({
        'Method': df['method'],
        'Accuracy': df['fold2.test'],
        'Fold': 'Fold 2 Test'
    })

    # Combining data for both folds
    data = pd.concat([data_fold1, data_fold2])

    # Order methods by accuracy
    method_order = data.groupby('Method')['Accuracy'].mean().sort_values(ascending=False).index

    # Convert Method column to categorical with the desired order
    data['Method'] = pd.Categorical(data['Method'], categories=method_order, ordered=True)

    # Plotting with different shapes for each algorithm
    plot_combined = (ggplot(data, aes(x='Accuracy', y='Method')) +
                    geom_point(size=1) +
                    labs(title="",
                        x="Accuracy Percentage",
                        y="Method") +
                    facet_wrap('~Fold', ncol=2) +
                    geom_vline(xintercept=90, color="black", size=1) +
                    theme_minimal() +
                    theme(legend_position='bottom', text=element_text(size=8)) +
                    theme(aspect_ratio=0.5)
                    )  # Adjust aspect ratio to decrease the distance between methods

    return plot_combined


if __name__ == "__main__":
    # save figure into pdf
    df = pd.read_csv(acc_rate_csv_path)
    plot_combined = acc_plot(df)
    plot_combined.save(figures_path + '/' + dataset_name + "_AccRateComparison" + ".pdf")
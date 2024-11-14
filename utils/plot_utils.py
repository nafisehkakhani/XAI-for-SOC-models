import os
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

# Set global preferences
plt.rcParams['font.family'] = "Calibri"  # Set font family to Times
plt.rcParams['font.size'] = 10  # Set default font size
plt.rcParams['axes.titlesize'] = 12  # Set title font size
plt.rcParams['axes.labelsize'] = 10  # Set xlabel and ylabel font size
plt.rcParams['xtick.labelsize'] = 10  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 10  # Set y-axis tick label size

# Set gridline settings
plt.rcParams['axes.grid'] = True  # Show gridlines
plt.rcParams['axes.grid.which'] = 'both'  # Show both major and minor gridlines
plt.rcParams['axes.grid.axis'] = 'both'  # Show gridlines on both axes
plt.rcParams['grid.color'] = 'white'  # Set gridline color
plt.rcParams['grid.linestyle'] = '--'  # Set gridline style
plt.rcParams['grid.linewidth'] = 1  # Set gridline width

# Set subplot border settings
plt.rcParams['axes.linewidth'] = 2  # Set subplot border linewidth
# Set background color
plt.rcParams['axes.facecolor'] = '#F0F0F0'  # Set background color of subplots
# plt.rcParams['axes.facecolor'] = 'white'  # Set background color of subplots

import pandas as pd

def find_highest_values(df):
    # Create empty lists to store highest values and corresponding column names
    highest_values = []
    second_highest_values = []
    third_highest_values = []
    corresponding_columns = []
    second_corresponding_columns = []
    third_corresponding_columns = []

    # Iterate over rows
    for index, row in df.iterrows():

        # Convert the values to numeric type
        numeric_row = pd.to_numeric(row)
        
        # Find the highest value and its corresponding column name for the current row
        sorted_values = numeric_row.sort_values(ascending=False)
        
        highest_value = sorted_values.iloc[0]
        second_highest_value = sorted_values.iloc[1]
        third_highest_value = sorted_values.iloc[2]
        
        highest_column = sorted_values.index[0]
        second_highest_column = sorted_values.index[1]
        third_highest_column = sorted_values.index[2]
        
        # Append the highest value and corresponding column name to the lists
        highest_values.append(highest_value)
        second_highest_values.append(second_highest_value)
        third_highest_values.append(third_highest_value)
        corresponding_columns.append(highest_column)
        second_corresponding_columns.append(second_highest_column)
        third_corresponding_columns.append(third_highest_column)

    # Create a DataFrame from the lists
    result_df = pd.DataFrame({
        'Highest_Value': highest_values,
        'Second_Highest_Value': second_highest_values,
        'Third_Highest_Value': third_highest_values,
        'Corresponding_Column': corresponding_columns,
        'Second_Corresponding_Column': second_corresponding_columns,
        'Third_Corresponding_Column': third_corresponding_columns
    })

    return result_df

# Example usage:
# result_df = find_highest_values(df)


class TextColors:
    """
    Source: https://github.com/moienr/TemporalGAN/blob/main/dataset/utils/utils.py
    A class containing ANSI escape codes for printing colored text to the terminal.
    
    Usage:
    ------
    ```
    print(TextColors.HEADER + 'This is a header' + TextColors.ENDC)
    print(TextColors.OKBLUE + 'This is OK' + TextColors.ENDC)
    ```
    
    Attributes:
    -----------
    `HEADER` : str
        The ANSI escape code for a bold magenta font color.
    `OKBLUE` : str
        The ANSI escape code for a bold blue font color.
    `OKCYAN` : str
        The ANSI escape code for a bold cyan font color.
    `OKGREEN` : str
        The ANSI escape code for a bold green font color.
    `WARNING` : str
        The ANSI escape code for a bold yellow font color.
    `FAIL` : str
        The ANSI escape code for a bold red font color.
    `ENDC` : str
        The ANSI escape code for resetting the font color to the default.
    `BOLD` : str
        The ANSI escape code for enabling bold font style.
    `UNDERLINE` : str
        The ANSI escape code for enabling underlined font style.
        
    Subclasses:
    `BOLDs`
    `UNDERLINEs`
    `BACKGROUNDs`
    `HIGHLIGHTs`
    `HIGH_INTENSITYs`
    `BOLD_HIGH_INTENSITYs`
    `HIGH_INTENSITY_BACKGROUNDs`
    `BOLD_BACKGROUNDs`
    
    
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    SLIME = '\033[38;2;165;165;0m'
    
    
    class BOLDs:
        BLACK = '\033[1;30m'
        RED = '\033[1;31m'
        GREEN = '\033[1;32m'
        YELLOW = '\033[1;33m'
        BLUE = '\033[1;34m'
        PURPLE = '\033[1;35m'
        CYAN = '\033[1;36m'
        WHITE = '\033[1;37m'
        ORANGE ='\033[38;2;255;165;0m'


    class UNDERLINEs:
        BLACK = '\033[4;30m'
        RED = '\033[4;31m'
        GREEN = '\033[4;32m'
        YELLOW = '\033[4;33m'
        BLUE = '\033[4;34m'
        PURPLE = '\033[4;35m'
        CYAN = '\033[4;36m'
        WHITE = '\033[4;37m'
    
    class BACKGROUNDs:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
        BLUE = '\033[44m'
        PURPLE = '\033[45m'
        CYAN = '\033[46m'
        WHITE = '\033[47m'
        DEFAULT = '\033[49m'
    
    class HIGH_INTENSITYs:
        BLACK = '\033[0;90m'
        RED = '\033[0;91m'
        GREEN = '\033[0;92m'
        YELLOW = '\033[0;93m'
        BLUE = '\033[0;94m'
        PURPLE = '\033[0;95m'
        CYAN = '\033[0;96m'
        WHITE = '\033[0;97m'
    
    class BOLD_HIGH_INTENSITYs:
        BLACK = '\033[1;90m'
        RED = '\033[1;91m'
        GREEN = '\033[1;92m'
        YELLOW = '\033[1;93m'
        BLUE = '\033[1;94m'
        PURPLE = '\033[1;95m'
        CYAN = '\033[1;96m'
        WHITE = '\033[1;97m'
        
    class HIGH_INTENSITY_BACKGROUNDs:
        BLACK = '\033[0;100m'
        RED = '\033[0;101m'
        GREEN = '\033[0;102m'
        YELLOW = '\033[0;103m'
        BLUE = '\033[0;104m'
        PURPLE = '\033[0;105m'
        CYAN = '\033[0;106m'
        WHITE = '\033[0;107m'

    class BOLD_BAKGROUNDs:
        BLACK = '\033[1;40m'
        RED = '\033[1;41m'
        GREEN = '\033[1;42m'
        YELLOW = '\033[1;43m'
        BLUE = '\033[1;44m'
        PURPLE = '\033[1;45m'
        CYAN = '\033[1;46m'
        WHITE = '\033[1;47m'
        ORANGE = '\033[48;2;255;165;0m\033[1m'
        S1 ='\033[48;2;100;50;50m'
        S2 = '\033[48;2;50;50;100m'
    
    class BLACK_TEXT_WIHT_BACKGROUNDs:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
        BLUE = '\033[44m'
        PURPLE = '\033[45m'
        CYAN = '\033[46m'
        WHITE = '\033[47m'
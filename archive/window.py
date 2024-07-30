import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from utils import get_stock_tickers, get_ticker_data, get_risk_contribution
from strategies import gmvp, mvp, ewp, ivp, rpp

class Window:
    def __init__(self, initial_amount=10000.) -> None:
        self.initial_amount = initial_amount
        
        self.root = tk.Tk()
        self.root.title("Stock Ticker and Date Selector")
        
        test_frame = ttk.Frame(self.root)
        test_frame.pack(side='top', fill='both', expand=True)
        
        test_frame.columnconfigure([0, 1, 2, 3], weight=1, uniform="column")
        for row_index in range(22):  # Ensure enough rows for all elements
            test_frame.rowconfigure(row_index, weight=1)

        self.init_date_widgets(test_frame)
        self.init_ticker_widgets(test_frame)
        self.init_strategy_dropdown(test_frame)
        self.init_portfolio_allocation_graph_area(test_frame)
        self.init_risk_contribution_graph_area(test_frame)
        
    def init_date_widgets(self, test_frame):
        start_date_label = ttk.Label(test_frame, text="Start Date")
        start_date_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.start_date_entry = DateEntry(test_frame, title_foreground='black')
        self.start_date_entry.bind("<<DateEntrySelected>>", self.run_strategy)
        self.start_date_entry.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        end_date_label = ttk.Label(test_frame, text="End Date")
        end_date_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.end_date_entry = DateEntry(test_frame, title_foreground='black')
        self.end_date_entry.bind("<<DateEntrySelected>>", self.run_strategy)
        self.end_date_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        
    def init_ticker_widgets(self, test_frame):
        ticker_label = ttk.Label(test_frame, text="Stock Tickers")
        ticker_label.grid(row=0, column=2, padx=10, pady=10, sticky="ew")
        self.ticker_listbox = tk.Listbox(test_frame, selectmode='multiple', height=6)
        tickers = sorted(get_stock_tickers())
        for ticker in tickers:
            self.ticker_listbox.insert(tk.END, ticker)
        self.ticker_listbox.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        
        show_button = ttk.Button(test_frame, text="Confirm Selected Tickers", command=self.display_selected_tickers)
        show_button.grid(row=2, column=2, padx=10, pady=10, sticky="ew")
        
        self.selected_tickers_label = ttk.Label(test_frame, text="Selected Tickers: None", wraplength=215)
        self.selected_tickers_label.grid(row=3, column=2, padx=10, pady=10, sticky="ew")

    def init_strategy_dropdown(self, test_frame):
        strategies_label = ttk.Label(test_frame, text="Investment Strategies")
        strategies_label.grid(row=0, column=3, padx=10, pady=10, sticky="ew")
        strategies = ['EWP', 'IVP', 'MVP', 'GMVP', 'RPP']
        self.category_combobox = ttk.Combobox(test_frame, values=strategies, state="readonly")
        self.category_combobox.grid(row=1, column=3, padx=10, pady=5, sticky="ew")
        self.category_combobox.bind("<<ComboboxSelected>>", self.run_strategy)

    def init_portfolio_allocation_graph_area(self, test_frame):
        # Initialize matplotlib figure with a smaller size
        self.port_fig, self.port_ax = plt.subplots(figsize=(1, 1))  # Width and height in inches
        self.port_ax.set_ylim(0, 1)
        self.port_canvas = FigureCanvasTkAgg(self.port_fig, master=test_frame)
        canvas_widget = self.port_canvas.get_tk_widget()
        canvas_widget.grid(row=4, column=0, columnspan=4, rowspan=8, sticky="nsew", padx=10, pady=10)
        
        # Adjust layout to add more space around the plot inside the canvas
        self.port_fig.subplots_adjust(left=0.1, right=0.95, top=0.75, bottom=0.25, hspace=0.55)
        
        self.port_ax.set_title("Portfolio Allocation")
        self.port_ax.set_xlabel("Tickers")
        self.port_ax.set_ylabel("Allocation (%)")
        
    def init_risk_contribution_graph_area(self, test_frame):
        # Initialize matplotlib figure with a smaller size
        self.risk_fig, self.risk_ax = plt.subplots(figsize=(1, 1))  # Width and height in inches
        self.risk_ax.set_ylim(0, 1)
        self.risk_canvas = FigureCanvasTkAgg(self.risk_fig, master=test_frame)
        canvas_widget = self.risk_canvas.get_tk_widget()
        canvas_widget.grid(row=13, column=0, columnspan=4, rowspan=8, sticky="nsew", padx=10, pady=10)
        
        # Adjust layout to add more space around the plot inside the canvas
        self.risk_fig.subplots_adjust(left=0.1, right=0.95, top=0.75, bottom=0.25)
        
        self.risk_ax.set_title("Risk Contribution")
        self.risk_ax.set_xlabel("Tickers")
        self.risk_ax.set_ylabel("Risk Contribution (%)")

    def display_selected_tickers(self):
        # Get currently selected indices
        selected_indices = self.ticker_listbox.curselection()

        # Display the tickers based on current selections
        self.selected_tickers = [self.ticker_listbox.get(i) for i in selected_indices]
        self.selected_tickers_label.config(text="Selected Tickers: " + ", ".join(self.selected_tickers))

        # Update the selections to be restored after button press
        self.selected_indices = selected_indices
        
        self.run_strategy()

    def run_strategy(self, event=None):
        selected_strategy = self.category_combobox.get()
        self.data = get_ticker_data(self.selected_tickers)

        if selected_strategy == 'EWP':
            output, weights = ewp(self.data, self.initial_amount)
        elif selected_strategy == 'IVP':
            output, weights = ivp(self.data, self.initial_amount)
        elif selected_strategy == 'MVP':
            output, weights = mvp(self.data, self.initial_amount, l=0.8)
        elif selected_strategy == 'GMVP':
            output, weights = gmvp(self.data, self.initial_amount)
        elif selected_strategy == 'RPP':
            output, weights = rpp(self.data, self.initial_amount)
            

        self.update_graph(weights)

        # Restore previously selected indices after updating the graph
        if hasattr(self, 'selected_indices'):
            self.ticker_listbox.selection_clear(0, tk.END)
            for i in self.selected_indices:
                self.ticker_listbox.selection_set(i)
            
    def update_graph(self, weights):
        # Clear the previous plot
        self.port_ax.clear()
        self.risk_ax.clear()
        
        risk_contribution = get_risk_contribution(weights, self.data)
        
        if isinstance(weights, pd.Series):
            weights.plot(kind='bar', ax=self.port_ax)
        elif isinstance(weights, pd.DataFrame):
            weights_column = weights['allocation']
            weights_column.sort_values(ascending=False).plot(kind='bar', ax=self.port_ax)
            
        if isinstance(risk_contribution, pd.Series):
            risk_contribution.plot(kind='bar', ax=self.risk_ax)
        elif isinstance(risk_contribution, pd.DataFrame):
            risk_contribution_column = risk_contribution['risk_contribution']
            risk_contribution_column.plot(kind='bar', ax=self.risk_ax)

        # Sort x-axis labels in alphabetical order
        self.port_ax.set_xticklabels(sorted(self.port_ax.get_xticklabels(), key=lambda x: x.get_text()))
        self.risk_ax.set_xticklabels(sorted(self.risk_ax.get_xticklabels(), key=lambda x: x.get_text()))
        
        for label in self.port_ax.get_xticklabels():
            label.set_rotation(0)
            
        for label in self.risk_ax.get_xticklabels():
            label.set_rotation(0)
        
        # Redraw the canvas to update the plot
        self.port_canvas.draw()
        self.risk_canvas.draw()

if __name__ == "__main__":
    window = Window()
    window.root.mainloop()

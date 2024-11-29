import React, { useState, useEffect } from 'react';
import DatePicker from 'react-datepicker';
import Select from 'react-select';
import { Bar } from 'react-chartjs-2';
import 'react-datepicker/dist/react-datepicker.css';
import axios from 'axios';


interface Option {
  value: string;
  label: string;
}

const strategies = [
  { value: 'ewp', label: 'Equal Weighted Portfolio' },
  { value: 'mvp', label: 'Minimum Variance Portfolio' },
  { value: 'ivp', label: 'Inverse Variance Portfolio' },
  { value: 'rpp', label: 'Risk Parity Portfolio' },
  { value: 'gmvp', label: 'Global Minimum Variance Portfolio' },
  { value: 'nrbp', label: 'Naive Risk Budget Portfolio' },
  { value: 'srp', label: 'Sharpe Ratio Portfolio' },
];

const modelTypes = [
  { value: 'capm', label: 'Capital Asset Pricing Model' },
  { value: 'ff3', label: 'Fama-French 3-Factor Model' },
  { value: 'ff5', label: 'Fama-French 5-Factor Model' },
];

const riskLevels = [
  { value: 'very_low', label: 'Very Low' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
  { value: 'very_high', label: 'Very High' },
];

const Form: React.FC = () => {
  const [startDate, setStartDate] = useState<Date | null>(null);
  const [endDate, setEndDate] = useState<Date | null>(null);
  const [amount, setAmount] = useState<number | string>('');

  const [selectedStockTickers, setSelectedStockTickers] = useState<Option[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Option | null>(null);
  const [selectedModelType, setSelectedModelType] = useState<Option | null>(null);
  const [stockOptions, setStockOptions] = useState<Option[]>([]);
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<Option | null>(null);

  const [startDateError, setStartDateError] = useState<string | null>(null);
  const [endDateError, setEndDateError] = useState<string | null>(null);
  const [dateError, setDateError] = useState<string | null>(null);
  const [amountError, setAmountError] = useState<string | null>(null);
  const [riskLevelError, setRiskLevelError] = useState<string | null>(null);
  const [tickersError, setTickersError] = useState<string | null>(null);
  const [strategyError, setStrategyError] = useState<string | null>(null);
  const [modelTypeError, setModelTypeError] = useState<string | null>(null);

  const [barChartData, setBarChartData] = useState<any>(null);
  const [plot1Url, setPlot1Url] = useState<string | null>(null);
  const [plot2Url, setPlot2Url] = useState<string | null>(null);



  useEffect(() => {
    axios.get<string[]>('http://127.0.0.1:8000/tickers')
      .then(response => {
        const formattedOptions = response.data.map(item => ({
          value: item,
          label: item,
        }));
        setStockOptions(formattedOptions);
      })
      .catch(error => {
        console.error('Error fetching tickers:', error);
      });
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
  
    // Reset error messages
    setStartDateError(null);
    setEndDateError(null);
    setAmountError(null);
    setTickersError(null);
    setStrategyError(null);
    setModelTypeError(null);
    setDateError(null);
    setRiskLevelError(null);
  
    // Validation checks
    let valid = true;
  
    if (!startDate) {
      setDateError('Start date is required.');
      valid = false;
    } else if (!endDate) {
      setDateError('End date is required.');
      valid = false;
    } else if (startDate >= endDate) {
      setDateError('Start date must be before end date.');
      valid = false;
    }
    if (!amount || parseFloat(amount.toString()) <= 0) {
      setAmountError('Initial amount must be greater than 0.');
      valid = false;
    }
    if (!selectedRiskLevel) {
      setRiskLevelError('Risk Level selection is required.');
      valid = false;
    }
    if (selectedStockTickers.length === 0) {
      setTickersError('At least one stock ticker must be selected.');
      valid = false;
    }
    if (!selectedStrategy) {
      setStrategyError('Strategy selection is required.');
      valid = false;
    }
    if (!selectedModelType) {
      setModelTypeError('Model type selection is required.');
      valid = false;
    }
  
    if (!valid) {
      return; // Exit if validation fails
    }
  
    // Prepare the data to send
    const formData = {
      start_date: startDate?.toISOString().split('T')[0], // Format date as 'YYYY-MM-DD'
      end_date: endDate?.toISOString().split('T')[0],     // Format date as 'YYYY-MM-DD'
      initial_amount: parseFloat(amount.toString()),      // Ensure amount is a float
      tickers: selectedStockTickers.map(ticker => ticker.value),
      strategy: selectedStrategy?.value,
      model_type: selectedModelType?.value,
      risk_level: selectedRiskLevel?.value,
    };
  
    try {
      // Send POST request to the backend
      const response = await axios.post('http://127.0.0.1:8000/portfolio-analysis', formData, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      console.log('Response:', response.data);
      // Handle the response as needed

      const { portfolio, plot1, plot2 } = response.data;
      setPlot1Url(`http://127.0.0.1:8000/${plot1}`);
      setPlot2Url(`http://127.0.0.1:8000/${plot2}`);

      // const responseData = response.data; // Assuming this is an object with Ticker: Amount

      // Prepare data for the bar chart
      const labels = Object.keys(portfolio);
      const data = Object.values(portfolio);

      setBarChartData({
        labels,
        datasets: [
          {
            label: 'Amount',
            data,
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
          },
        ],
      });
    } catch (error) {
      console.error('Error submitting form:', error);
      // Handle error as needed
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex space-x-4">
        <div className="flex-1">
          <label className="block">Start Date:</label>
          <DatePicker
            selected={startDate}
            onChange={(date: Date | null) => setStartDate(date)}
            className="border p-2 w-full"
          />
          {startDateError && <p className="text-red-500">{startDateError}</p>}
        </div>
        <div className="flex-1">
          <label className="block">End Date:</label>
          <DatePicker
            selected={endDate}
            onChange={(date: Date | null) => setEndDate(date)}
            className="border p-2 w-full"
          />
          {endDateError && <p className="text-red-500">{endDateError}</p>}
        </div>
        <div className="flex-1">
          <label className="block">Initial Amount:</label>
          <input
            type="number"
            value={amount}
            onChange={(e) => {
              const value = Math.max(0, parseFloat(e.target.value));
              setAmount(value);
            }}
            className="border p-2 w-full"
          />
          {amountError && <p className="text-red-500">{amountError}</p>}
        </div>
        <div className="flex-1">
          <label className="block">Risk Level:</label>
          <Select
            options={riskLevels}
            value={selectedRiskLevel}
            onChange={(selected) => setSelectedRiskLevel(selected as Option)}
            className="border p-2 w-full"
          />
          {riskLevelError && <p className="text-red-500">{riskLevelError}</p>}
        </div>
      </div>
      <div className="flex space-x-4">
        <div className="flex-1">
          <label className="block">Stock Tickers:</label>
          <Select
            isMulti
            options={stockOptions}
            value={selectedStockTickers}
            onChange={(selected) => setSelectedStockTickers(selected as Option[])}
            className="border p-2 w-full"
          />
          {tickersError && <p className="text-red-500">{tickersError}</p>}
        </div>
        <div className="flex-1">
          <label className="block">Strategy:</label>
          <Select
            options={strategies}
            value={selectedStrategy}
            onChange={(selected) => setSelectedStrategy(selected as Option)}
            className="border p-2 w-full"
          />
          {strategyError && <p className="text-red-500">{strategyError}</p>}
        </div>
        <div className="flex-1">
          <label className="block">Model Type:</label>
          <Select
            options={modelTypes}
            value={selectedModelType}
            onChange={(selected) => setSelectedModelType(selected as Option)}
            className="border p-2 w-full"
          />
          {modelTypeError && <p className="text-red-500">{modelTypeError}</p>}
        </div>
      </div>
      <button type="submit" className="bg-blue-500 text-white p-2 mt-4">
        Submit
      </button>
      {barChartData && (
        <div className="bar-chart-container">
          {/* <h2 className="text-center">Portfolio Allocation</h2> */}
          <Bar
            data={barChartData}
            options={{
              responsive: true,
              plugins: {
                legend: {
                  position: 'top',
                },
                title: {
                  display: true,
                  text: 'Portfolio Allocation by Ticker',
                },
              },
            }}
          />
        </div>
      )}

      {plot1Url && plot2Url && (
        <div className="plots-container">
          <img src={plot1Url} alt="Historical and Forecasted Portfolio Returns" />
          <img src={plot2Url} alt="Historical and Forecasted Portfolio Value" />
        </div>
      )}

    </form>

  );
};

export default Form;

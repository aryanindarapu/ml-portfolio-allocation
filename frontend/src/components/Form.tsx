import React, { useState, useEffect } from 'react';
import DatePicker from 'react-datepicker';
import Select from 'react-select';
import { Bar, Scatter } from 'react-chartjs-2';
import 'react-datepicker/dist/react-datepicker.css';
import axios from 'axios';


interface Option {
  value: string;
  label: string;
}

const strategies = [
  { value: 'mvp', label: 'Standard' },
  { value: 'patv1', label: 'Transformer (Beta)' },
];

const riskLevels = [
  { value: 'very_low', label: 'Very Low' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
  { value: 'very_high', label: 'Very High' },
];

const Form: React.FC = () => {
  const [amount, setAmount] = useState<number | string>('');

  const [selectedStockTickers, setSelectedStockTickers] = useState<Option[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Option | null>(null);
  const [runSimulation, setRunSimulation] = useState<boolean>(false);
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<Option | null>(null);
  const [stockOptions, setStockOptions] = useState<Option[]>([]);

  const [amountError, setAmountError] = useState<string | null>(null);
  const [riskLevelError, setRiskLevelError] = useState<string | null>(null);
  const [tickersError, setTickersError] = useState<string | null>(null);
  const [strategyError, setStrategyError] = useState<string | null>(null);

  const [barChartData, setBarChartData] = useState<any>(null);
  const [scatterChartData, setScatterChartData] = useState<any>(null);

  const [loading, setLoading] = useState<boolean>(false);

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

  useEffect(() => {
    handleSubmit();
  }, [amount, selectedStockTickers, selectedStrategy, selectedRiskLevel]);

  // const handleSubmit = async (e: React.FormEvent) => {
  const handleSubmit = async () => {
    // if (!amount || parseFloat(amount.toString()) <= 0 || selectedStockTickers.length < 2 || !selectedStrategy || !selectedRiskLevel) {
    //   return;
    // }
    let valid = true;
    if (!amount || parseFloat(amount.toString()) <= 0) {
      setAmountError('Please enter a valid amount');
      valid = false;
    } else {
      setAmountError(null);
    }

    if (selectedStockTickers.length < 2) {
      setTickersError('Please select at least two stock tickers');
      valid = false;
    } else {
      setTickersError(null);
    }

    if (!selectedStrategy) {
      setStrategyError('Please select a strategy');
      valid = false;
    } else {
      setStrategyError(null);
    }

    if (!selectedRiskLevel) {
      setRiskLevelError('Please select a risk level');
      valid = false;
    } else {
      setRiskLevelError(null);
    }

    if (!valid) {
      return;
    }

    setLoading(true);
    const minLoadingTime = 1500;
    const loadingStartTime = Date.now();
  
    const formData = {
      start_date: '2010-01-01',
      end_date: new Date().toISOString().split('T')[0],
      initial_amount: parseFloat(amount.toString()),
      tickers: selectedStockTickers.map((ticker) => ticker.value),
      strategy: selectedStrategy?.value,
      risk_level: selectedRiskLevel?.value,
    };
  
    try {
      const response = await axios.post('http://127.0.0.1:8000/portfolio-analysis', formData, {
        headers: { 'Content-Type': 'application/json' },
      });

      const elapsedTime = Date.now() - loadingStartTime;
      const remainingTime = Math.max(0, minLoadingTime - elapsedTime);

      setTimeout(() => {
        console.log(response.data);
        const responseData = response.data;

        const labels = Object.keys(responseData.weights);
        const data = Object.values(responseData.weights).map((val) => Math.round((val as number) * 100) / 100);

        setBarChartData({
          labels,
          datasets: [
            {
              label: 'Amount',
              data,
              backgroundColor: 'rgba(109, 40, 217, 0.3)',
              borderColor: 'rgba(109, 40, 217, 1)',
              borderWidth: 1,
            },
          ],
        });
        
        const scatterData = responseData.frontier; // Assuming scatter_data is the (3, n) numpy array from backend
        const dataPoints = scatterData[0].map((_: any, i: number) => ({
          x: scatterData[1][i], // Risk (x-axis)
          y: scatterData[0][i], // Return (y-axis)
          r: 3, // Optional: size of the point
          sharpe: scatterData[2][i], // Sharpe ratio (color value)
        }));
        
        // setScatterChartData({
        //   datasets: [
        //     {
        //       label: 'Efficient Frontier',
        //       data: dataPoints,
        //       backgroundColor: dataPoints.map((point: any) =>
        //         `rgba(${Math.min(255, point.sharpe * 50)}, ${Math.max(255 - point.sharpe * 50, 0)}, 150, 0.6)`
        //       ),
        //     },
        //   ],
        // });
        // Function to interpolate between two RGB colors
        const interpolateColor = (startColor: [number, number, number], endColor: [number, number, number], ratio: number) => {
          const r = Math.round(startColor[0] + (endColor[0] - startColor[0]) * ratio);
          const g = Math.round(startColor[1] + (endColor[1] - startColor[1]) * ratio);
          const b = Math.round(startColor[2] + (endColor[2] - startColor[2]) * ratio);
          return `rgba(${r}, ${g}, ${b}, 0.6)`; // Alpha set to 0.6 for transparency
        };

        // Define your start and end colors
        const startColor: [number, number, number] = [255, 255, 0]; // Yellow (R, G, B)
        const endColor: [number, number, number] = [109, 40, 217];

        // Generate the scatter chart dataset
        setScatterChartData({
          datasets: [
            {
              label: 'Efficient Frontier',
              data: dataPoints,
              backgroundColor: dataPoints.map((point: any) =>
                interpolateColor(startColor, endColor, Math.min(1, Math.max(0, point.sharpe))) // Clamp ratio to [0, 1]
              ),
            },
          ],
        });

        
        setLoading(false);
      }, remainingTime);
    } catch (error) {
      console.error('Error submitting form:', error);
      setLoading(false);
    }
  };
    
  return (
    <form className="space-y-4">
      <div className="flex space-x-4">
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
          <label className="block mb-2">Run Simulation:</label>
          <div
            className={`relative w-12 h-6 flex items-center bg-gray-300 rounded-full cursor-pointer ${
              runSimulation ? 'bg-violet-700' : 'bg-gray-300'
            }`}
            onClick={() => setRunSimulation(!runSimulation)}
          >
            <div
              className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${
                runSimulation ? 'translate-x-6' : 'translate-x-0'
              }`}
            ></div>
          </div>
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
      </div>
      {loading && (
        <div className="flex justify-center items-center">
          <div className="animate-spin rounded-full h-8 w-8 border-t-4 border-violet-500 border-solid border-opacity-50"></div>
          <p className="ml-3 text-center">Loading...</p>
        </div>
        
      )}
      {barChartData && (
        <div className="bar-chart-container">
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
      {runSimulation && scatterChartData && (
        <div className="scatter-chart-container mt-8">
          <Scatter
            data={scatterChartData}
            options={{
              responsive: true,
              scales: {
                x: {
                  title: { display: true, text: 'Risk (Volatility)' },
                },
                y: {
                  title: { display: true, text: 'Return' },
                },
              },
              plugins: {
                datalabels: {
                  display: false,
                },
                tooltip: {
                  callbacks: {
                    label: (context) => {
                      const point: any = context.raw;
                      return `Return: ${point.y}, Risk: ${point.x}, Sharpe: ${point.sharpe.toFixed(2)}`;
                    },
                  },
                },
                title: {
                  display: true,
                  text: 'Efficient Frontier',
                },
                legend: {
                  display: false,
                },
              },
            }}
          />
        </div>
      )}

    </form>

  );
};

export default Form;

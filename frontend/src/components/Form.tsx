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

function reformatForHtmlOverlay(text: string) {
  // Convert markdown bold markers to <strong> tags.
  let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Replace newline characters with <br> tags.
  html = html.replace(/\n/g, '<br>');
  return html;
}

const Form: React.FC = () => {
  const [amount, setAmount] = useState<number | string>('');
  const [selectedStockTickers, setSelectedStockTickers] = useState<Option[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Option | null>(null);
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<Option | null>(null);
  const [stockOptions, setStockOptions] = useState<Option[]>([]);
  
  // Errors
  const [amountError, setAmountError] = useState<string | null>(null);
  const [riskLevelError, setRiskLevelError] = useState<string | null>(null);
  const [tickersError, setTickersError] = useState<string | null>(null);
  const [strategyError, setStrategyError] = useState<string | null>(null);
  
  // Chart Data
  const [barChartData, setBarChartData] = useState<any>(null);
  const [scatterChartData, setScatterChartData] = useState<any>(null);
  
  // Loading / Overlay state
  const [loading, setLoading] = useState<boolean>(false);
  const [overlayVisible, setOverlayVisible] = useState<boolean>(false);
  const [overlayContent, setOverlayContent] = useState<string>('');
  const [runSimulation, setRunSimulation] = useState<boolean>(false);

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

  // Update the overlay handler to call the complex workflow endpoint
  const handleOpenOverlay = async () => {
    setOverlayVisible(true);
    setOverlayContent('Loading data...');
    
    const formData = {
      start_date: '2010-01-01',
      end_date: new Date().toISOString().split('T')[0],
      initial_amount: parseFloat(amount.toString()),
      tickers: selectedStockTickers.map((ticker) => ticker.value),
      strategy: selectedStrategy?.value,
      risk_level: selectedRiskLevel?.value,
    };

    try {
      const response = await axios.post('http://127.0.0.1:8000/run-complex-workflow', formData, {
        headers: { 'Content-Type': 'application/json' },
      });
      const formattedOutput = reformatForHtmlOverlay(response.data.final_output);
      setOverlayContent(formattedOutput);
    } catch (error) {
      console.error('Error fetching overlay data:', error);
      setOverlayContent("Error fetching data. Please try again.");
    }
  };

  // Form submission for portfolio analysis (for the charts)
  const handleSubmit = async () => {
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
        const responseData = response.data;

        // Prepare bar chart data (portfolio allocation)
        const labels = Object.keys(responseData.weights);
        const data = Object.values(responseData.weights).map((val) =>
          Math.round((val as number) * 100) / 100
        );

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

        // Prepare scatter chart data (efficient frontier)
        const scatterData = responseData.frontier;
        const dataPoints = scatterData[0].map((_: any, i: number) => ({
          x: scatterData[1][i], // Return on x-axis
          y: scatterData[0][i], // Volatility on y-axis
          r: 3,               // Optional: point size
          sharpe: scatterData[2][i], // Sharpe ratio
        }));

        // Interpolate colors for scatter chart points
        const interpolateColor = (startColor: [number, number, number], endColor: [number, number, number], ratio: number) => {
          const r = Math.round(startColor[0] + (endColor[0] - startColor[0]) * ratio);
          const g = Math.round(startColor[1] + (endColor[1] - startColor[1]) * ratio);
          const b = Math.round(startColor[2] + (endColor[2] - startColor[2]) * ratio);
          return `rgba(${r}, ${g}, ${b}, 0.6)`;
        };

        const startColor: [number, number, number] = [255, 255, 0]; // Yellow
        const endColor: [number, number, number] = [109, 40, 217];

        setScatterChartData({
          datasets: [
            {
              label: 'Efficient Frontier',
              data: dataPoints,
              backgroundColor: dataPoints.map((point: any) =>
                interpolateColor(startColor, endColor, Math.min(1, Math.max(0, point.sharpe)))
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
            className={`relative w-12 h-6 flex items-center rounded-full cursor-pointer ${
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
      <div className="flex space-x-4 items-center">
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
        <div className="flex-2 flex items-center">
          <button
            type="button"
            className="bg-violet-700 text-white p-2 rounded"
            onClick={handleOpenOverlay}
          >
            What does this mean?
          </button>
        </div>
      </div>

      {overlayVisible && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-white p-4 rounded shadow-lg relative max-w-[80vw] max-h-[80vh] overflow-auto">
            <button
              className="absolute top-2 right-2 text-gray-500"
              onClick={() => setOverlayVisible(false)}
            >
              &times;
            </button>
            <div className="p-4">
              <h2 className="text-xl font-bold mb-2">Portfolio Insights</h2>
              <p dangerouslySetInnerHTML={{ __html: overlayContent }}></p>
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex justify-center items-center">
          <div className="animate-spin rounded-full h-8 w-8 border-t-4 border-violet-500"></div>
          <p className="ml-3">Loading...</p>
        </div>
      )}
      
      {barChartData && (
        <div className="bar-chart-container">
          <Bar
            data={barChartData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Portfolio Allocation by Ticker' },
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
                x: { title: { display: true, text: 'Risk (Volatility)' } },
                y: { title: { display: true, text: 'Return' } },
              },
              plugins: {
                tooltip: {
                  callbacks: {
                    label: (context) => {
                      const point: any = context.raw;
                      return `Return: ${point.y}, Risk: ${point.x}, Sharpe: ${point.sharpe.toFixed(2)}`;
                    },
                  },
                },
                title: { display: true, text: 'Efficient Frontier' },
                legend: { display: false },
              },
            }}
          />
        </div>
      )}
    </form>
  );
};

export default Form;

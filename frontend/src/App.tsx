import React from 'react';
import Form from './components/Form';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  ChartDataLabels
);

const App: React.FC = () => {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Portfolio Allocation</h1>
      <main>
        <Form />
      </main>
    </div>
  );
};

export default App;

// import React, { useState } from 'react';
// import Form from './components/Form';
// import Charts from './components/Charts';

// const App: React.FC = () => {
//   const [chartData, setChartData] = useState<{ [key: string]: number } | null>(null);

//   const handleFormSubmit = (data: { [key: string]: number }) => {
//     setChartData(data);
//   };

//   return (
//     <div className="container mx-auto p-4">
//       <h1 className="text-2xl font-bold mb-4">React TypeScript App</h1>
//       <Form onSubmit={handleFormSubmit} />
//       <div className="mt-8">
//         <Charts data={chartData} />
//       </div>
//     </div>
//   );
// };

// export default App;

import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const SentimentChart = ({ history, symbol }) => {
  if (!history || history.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <p>No historical data available yet.</p>
        <p className="text-sm mt-2">Analyze this stock again to see trends over time.</p>
      </div>
    );
  }

  // Sort by date (oldest first)
  const sortedHistory = [...history].sort((a, b) => 
    new Date(a.timestamp) - new Date(b.timestamp)
  );

  const labels = sortedHistory.map((entry, index) => {
    const date = new Date(entry.timestamp);
    // Show date in a readable format
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  });

  const sentimentData = sortedHistory.map(entry => entry.sentiment);
  const confidenceData = sortedHistory.map(entry => entry.confidence);

  const data = {
    labels,
    datasets: [
      {
        label: 'Sentiment Score',
        data: sentimentData,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
        yAxisID: 'y',
      },
      {
        label: 'Confidence',
        data: confidenceData,
        borderColor: 'rgb(234, 179, 8)',
        backgroundColor: 'rgba(234, 179, 8, 0.1)',
        fill: true,
        tension: 0.4,
        yAxisID: 'y1',
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'rgb(209, 213, 219)',
        },
      },
      title: {
        display: true,
        text: `${symbol} Sentiment Trend`,
        color: 'rgb(209, 213, 219)',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
      tooltip: {
        backgroundColor: 'rgba(31, 41, 55, 0.9)',
        titleColor: 'rgb(209, 213, 219)',
        bodyColor: 'rgb(209, 213, 219)',
        borderColor: 'rgb(75, 85, 99)',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
        },
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Sentiment Score',
          color: 'rgb(209, 213, 219)',
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
        },
        min: -0.5,
        max: 0.5,
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Confidence',
          color: 'rgb(209, 213, 219)',
        },
        grid: {
          drawOnChartArea: false,
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
        },
        min: 0,
        max: 1,
      },
    },
  };

  return (
    <div className="w-full" style={{ height: '400px' }}>
      <Line data={data} options={options} />
    </div>
  );
};

export default SentimentChart;


import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  TimeScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  TimeScale,
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

  // Create data points with timestamps for proper time-based spacing
  const sentimentData = sortedHistory.map(entry => ({
    x: new Date(entry.timestamp).getTime(), // Use timestamp for x-axis
    y: entry.sentiment,
    timestamp: entry.timestamp, // Store timestamp for tooltip
    sentiment: entry.sentiment,
    confidence: entry.confidence
  }));

  const confidenceData = sortedHistory.map(entry => ({
    x: new Date(entry.timestamp).getTime(), // Use timestamp for x-axis
    y: entry.confidence,
    timestamp: entry.timestamp, // Store timestamp for tooltip
    sentiment: entry.sentiment,
    confidence: entry.confidence
  }));

  const data = {
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
        callbacks: {
          title: function(context) {
            // Show formatted date and time in tooltip title
            if (context.length > 0 && context[0].raw.timestamp) {
              const date = new Date(context[0].raw.timestamp);
              const dateStr = date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
              });
              const timeStr = date.toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
                hour12: true
              });
              return `${dateStr}, ${timeStr}`;
            }
            return '';
          },
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            
            // Format the value based on dataset
            let formattedValue;
            if (label === 'Sentiment Score') {
              formattedValue = value.toFixed(3);
            } else {
              formattedValue = (value * 100).toFixed(1) + '%';
            }
            
            return `${label}: ${formattedValue}`;
          }
        }
      },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
          displayFormats: {
            day: 'MMM d',
            hour: 'MMM d, h:mm a',
            minute: 'MMM d, h:mm a'
          },
          tooltipFormat: 'MMM d, yyyy h:mm a'
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
          maxRotation: 45,
          minRotation: 45,
        },
        title: {
          display: false
        }
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
          stepSize: 0.2,
        },
        min: -1.2,
        max: 1.2,
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
          stepSize: 0.1,
        },
        min: 0,
        max: 1,
      },
    },
  };

  return (
    <div className="w-full" style={{ height: '500px' }}>
      <Line data={data} options={options} />
    </div>
  );
};

export default SentimentChart;


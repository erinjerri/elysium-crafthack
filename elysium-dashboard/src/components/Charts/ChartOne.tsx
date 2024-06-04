import { ApexOptions } from 'apexcharts';
import React, { useState } from 'react';
import ReactApexChart from 'react-apexcharts';

const options: ApexOptions = {
  legend: {
    show: false,
    position: 'top',
    horizontalAlign: 'left',
  },
  colors: ['#3C50E0', '#80CAEE'],
  chart: {
    fontFamily: 'Satoshi, sans-serif',
    height: 335,
    type: 'area',
    dropShadow: {
      enabled: true,
      color: '#623CEA14',
      top: 10,
      blur: 4,
      left: 0,
      opacity: 0.1,
    },

    toolbar: {
      show: false,
    },
  },
  responsive: [
    {
      breakpoint: 1024,
      options: {
        chart: {
          height: 300,
        },
      },
    },
    {
      breakpoint: 1366,
      options: {
        chart: {
          height: 350,
        },
      },
    },
  ],
  stroke: {
    width: [2, 2],
    curve: 'straight',
  },
  // labels: {
  //   show: false,
  //   position: "top",
  // },
  grid: {
    xaxis: {
      lines: {
        show: true,
      },
    },
    yaxis: {
      lines: {
        show: true,
      },
    },
  },
  dataLabels: {
    enabled: false,
  },
  markers: {
    size: 4,
    colors: '#fff',
    strokeColors: ['#3056D3', '#80CAEE'],
    strokeWidth: 3,
    strokeOpacity: 0.9,
    strokeDashArray: 0,
    fillOpacity: 1,
    discrete: [],
    hover: {
      size: undefined,
      sizeOffset: 5,
    },
  },
  xaxis: {
    type: 'category',
    categories: [
      'Day 1',
      'Day 2',
      'Day 3',
      'Day 4',
      'Day 5',
      'Day 6',
      'Day 7',
      'Day 8',
      'Day 9',
      'Day 10',
      'Day 11',
      'Day 12',
      'Day 13', 
      'Day 14',
      'Day 15',
      'Day 16',
      'Day 17',
      'Day 18',
      'Day 19',
      'Day 20',
      'Day 21',
      'Day 22',
      'Day 23',
      'Day 24',
      'Day 25',
      'Day 26',
      'Day 27', 
      'Day 28',
      'Day 29',
      'Day 30',
      'Day 21',
      'Day 13', 
    ],
    axisBorder: {
      show: false,
    },
    axisTicks: {
      show: false,
    },
  },
  yaxis: {
    title: {
      style: {
        fontSize: '0px',
      },
    },
    min: 0,
    max: 100,
  },
};

interface ChartOneState {
  series: {
    name: string;
    data: number[];
  }[];
}

const ChartOne: React.FC = () => {
  const [state, setState] = useState<ChartOneState>({
    series: [
      {
        name: 'Sleep Duration Data',
        data: [
      {"date": "2024-05-01", "sleep_duration_hours": 7.5},
      {"date": "2024-05-02", "sleep_duration_hours": 6.8},
      {"date": "2024-05-03", "sleep_duration_hours": 8.2},
      {"date": "2024-05-04", "sleep_duration_hours": 7.0},
      {"date": "2024-05-05", "sleep_duration_hours": 7.3},
      {"date": "2024-05-06", "sleep_duration_hours": 6.5},
      {"date": "2024-05-07", "sleep_duration_hours": 7.8},
      {"date": "2024-05-08", "sleep_duration_hours": 8.0},
      {"date": "2024-05-09", "sleep_duration_hours": 6.9},
      {"date": "2024-05-10", "sleep_duration_hours": 7.2},
      {"date": "2024-05-11", "sleep_duration_hours": 7.6},
      {"date": "2024-05-12", "sleep_duration_hours": 6.7},
      {"date": "2024-05-13", "sleep_duration_hours": 7.9},
      {"date": "2024-05-14", "sleep_duration_hours": 7.1},
      {"date": "2024-05-15", "sleep_duration_hours": 8.3},
      {"date": "2024-05-16", "sleep_duration_hours": 6.4},
      {"date": "2024-05-17", "sleep_duration_hours": 7.5},
      {"date": "2024-05-18", "sleep_duration_hours": 8.1},
      {"date": "2024-05-19", "sleep_duration_hours": 7.0},
      {"date": "2024-05-20", "sleep_duration_hours": 6.6},
      {"date": "2024-05-21", "sleep_duration_hours": 7.7},
      {"date": "2024-05-22", "sleep_duration_hours": 8.4},
      {"date": "2024-05-23", "sleep_duration_hours": 6.8},
      {"date": "2024-05-24", "sleep_duration_hours": 7.3},
      {"date": "2024-05-25", "sleep_duration_hours": 7.9},
      {"date": "2024-05-26", "sleep_duration_hours": 6.5},
      {"date": "2024-05-27", "sleep_duration_hours": 8.0},
      {"date": "2024-05-28", "sleep_duration_hours": 7.2},
      {"date": "2024-05-29", "sleep_duration_hours": 6.9},
      {"date": "2024-05-30", "sleep_duration_hours": 8.1} ]

  },
      {
        name: 'Sleep Consistency',
        data: [
            {"date": "2024-05-01", "sleep_consistency_score": 8},
            {"date": "2024-05-02", "sleep_consistency_score": 6},
            {"date": "2024-05-03", "sleep_consistency_score": 9},
            {"date": "2024-05-04", "sleep_consistency_score": 7},
            {"date": "2024-05-05", "sleep_consistency_score": 7},
            {"date": "2024-05-06", "sleep_consistency_score": 5},
            {"date": "2024-05-07", "sleep_consistency_score": 8},
            {"date": "2024-05-08", "sleep_consistency_score": 9},
            {"date": "2024-05-09", "sleep_consistency_score": 6},
            {"date": "2024-05-10", "sleep_consistency_score": 7},
            {"date": "2024-05-11", "sleep_consistency_score": 8},
            {"date": "2024-05-12", "sleep_consistency_score": 6},
            {"date": "2024-05-13", "sleep_consistency_score": 9},
            {"date": "2024-05-14", "sleep_consistency_score": 7},
            {"date": "2024-05-15", "sleep_consistency_score": 9},
            {"date": "2024-05-16", "sleep_consistency_score": 5},
            {"date": "2024-05-17", "sleep_consistency_score": 8},
            {"date": "2024-05-18", "sleep_consistency_score": 9},
            {"date": "2024-05-19", "sleep_consistency_score": 7},
            {"date": "2024-05-20", "sleep_consistency_score": 5},
            {"date": "2024-05-21", "sleep_consistency_score": 8},
            {"date": "2024-05-22", "sleep_consistency_score": 10},
            {"date": "2024-05-23", "sleep_consistency_score": 6},
            {"date": "2024-05-24", "sleep_consistency_score": 7},
            {"date": "2024-05-25", "sleep_consistency_score": 9},
            {"date": "2024-05-26", "sleep_consistency_score": 5},
            {"date": "2024-05-27", "sleep_consistency_score": 9},
            {"date": "2024-05-28", "sleep_consistency_score": 7},
            {"date": "2024-05-29", "sleep_consistency_score": 6},
            {"date": "2024-05-30", "sleep_consistency_score": 9}
          
        ],
      },
    ],
  });

  const handleReset = () => {
    setState((prevState) => ({
      ...prevState,
    }));
  };
  handleReset;

  return (
    <div className="col-span-12 rounded-sm border border-stroke bg-white px-5 pt-7.5 pb-5 shadow-default dark:border-strokedark dark:bg-boxdark sm:px-7.5 xl:col-span-8">
      <div className="flex flex-wrap items-start justify-between gap-3 sm:flex-nowrap">
        <div className="flex w-full flex-wrap gap-3 sm:gap-5">
          <div className="flex min-w-47.5">
            <span className="mt-1 mr-2 flex h-4 w-full max-w-4 items-center justify-center rounded-full border border-primary">
              <span className="block h-2.5 w-full max-w-2.5 rounded-full bg-primary"></span>
            </span>
            <div className="w-full">
              <p className="font-semibold text-primary">Sleep Duration</p>
              <p className="text-sm font-medium">6-8 hours</p>
            </div>
          </div>
          <div className="flex min-w-47.5">
            <span className="mt-1 mr-2 flex h-4 w-full max-w-4 items-center justify-center rounded-full border border-secondary">
              <span className="block h-2.5 w-full max-w-2.5 rounded-full bg-secondary"></span>
            </span>
            <div className="w-full">
              <p className="font-semibold text-secondary">Sleep Consistency Sales</p>
              <p className="text-sm font-medium">1-10</p>
            </div>
          </div>
        </div>
        <div className="flex w-full max-w-45 justify-end">
          <div className="inline-flex items-center rounded-md bg-whiter p-1.5 dark:bg-meta-4">
            <button className="rounded bg-white py-1 px-3 text-xs font-medium text-black shadow-card hover:bg-white hover:shadow-card dark:bg-boxdark dark:text-white dark:hover:bg-boxdark">
              Day
            </button>
            <button className="rounded py-1 px-3 text-xs font-medium text-black hover:bg-white hover:shadow-card dark:text-white dark:hover:bg-boxdark">
              Week
            </button>
            <button className="rounded py-1 px-3 text-xs font-medium text-black hover:bg-white hover:shadow-card dark:text-white dark:hover:bg-boxdark">
              Month
            </button>
          </div>
        </div>
      </div>

      <div>
        <div id="chartOne" className="-ml-5">
          <ReactApexChart
            options={options}
            series={state.series}
            type="area"
            height={350}
          />
        </div>
      </div>
    </div>
  );
};

export default ChartOne;

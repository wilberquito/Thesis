<script lang="ts">
  import Chart from 'chart.js/auto';
  import { onMount } from 'svelte';
  import type { ChartConfiguration } from 'chart.js/auto'

  let chart: Chart;

  const data = {
    labels: [
      'Eating',
      'Drinking',
      'Sleeping',
      'Designing',
      'Coding',
      'Cycling',
      'Running'
    ],
    datasets: [{
      label: '',
      data: [65, 59, 90, 81, 56, 55, 40],
      fill: true,
      backgroundColor: 'rgba(255, 99, 132, 0.2)',
      borderColor: 'rgb(255, 99, 132)',
      pointBackgroundColor: 'rgb(255, 99, 132)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgb(255, 99, 132)'
    }, {
      label: '',
      data: [28, 48, 40, 19, 96, 27, 100],
      fill: true,
      backgroundColor: 'rgba(54, 162, 235, 0.2)',
      borderColor: 'rgb(54, 162, 235)',
      pointBackgroundColor: 'rgb(54, 162, 235)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgb(54, 162, 235)'
    }]
  };

  const config: ChartConfiguration<'radar'> = {
    type: 'radar',
    data: data,
    options: {
      scales: {
        r: {
          ticks: {
            display: false
          }
        }
      },
      elements: {
        line: {
          borderWidth: 3
        }
      }
    },
  };

  function initChart() {
    const canvas = document.getElementById('radarChart') as HTMLCanvasElement;
    // const ctx = canvas?.getContext('2d');
    chart = new Chart(canvas, config);
  }

  onMount(initChart);

</script>

  <canvas id="radarChart"></canvas>

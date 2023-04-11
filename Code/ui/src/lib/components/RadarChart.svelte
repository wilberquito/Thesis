<script lang="ts">
  import Chart from 'chart.js/auto';
  import { onMount } from 'svelte';
  import type { ChartConfiguration } from 'chart.js/auto'

  export let inputs = {}
  let chart: Chart;

  Chart.defaults.backgroundColor = '#9BD0F5';
  Chart.defaults.borderColor = '#1779ba';
  Chart.defaults.color = '#000';

  function initChart() {
    const labels = (Object.keys(inputs) as string[]).map(str => str.toUpperCase())
    const values = (Object.values(inputs) as number[]).map(n => Number((n*100).toFixed(2)))

    const data = {
      labels: labels,
      datasets: [{
        label: '',
        data: values,
        fill: true,
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgb(255, 99, 132)',
        pointBackgroundColor: 'rgb(255, 99, 132)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgb(255, 99, 132)',
        pointStyle: 'circle',
        pointHoverRadius: 6,
        pointRadius: 5,
      }]
    };

    const config: ChartConfiguration<'radar'> = {
      type: 'radar',
      data: data,
      options: {
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          r: {
            ticks: {
              display: false
            },
            suggestedMax: 100,
            beginAtZero: true
          }
        },
        elements: {
          line: {
            borderWidth: 3
          }
        }
      },
    };

    const canvas = document.getElementById('radarChart') as HTMLCanvasElement;
    // const ctx = canvas?.getContext('2d');
    chart = new Chart(canvas, config);
  }

  onMount(initChart);

</script>

  <canvas id="radarChart"></canvas>

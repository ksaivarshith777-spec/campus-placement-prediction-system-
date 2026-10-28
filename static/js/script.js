document.addEventListener('DOMContentLoaded', function () {
  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (navToggle && navLinks) {
    navToggle.addEventListener('click', function () {
      const isOpen = navLinks.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', String(isOpen));
    });

    navLinks.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        navLinks.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  const predictionForm = document.querySelector('.predict-form');
  if (predictionForm) {
    predictionForm.addEventListener('submit', function (event) {
      const fields = predictionForm.querySelectorAll('input');
      let hasInvalidValue = false;

      fields.forEach(function (field) {
        const value = field.value.trim();
        if (!value) {
          hasInvalidValue = true;
          return;
        }

        const numericValue = Number(value);
        if (Number.isNaN(numericValue)) {
          hasInvalidValue = true;
        }
      });

      if (hasInvalidValue) {
        event.preventDefault();
        alert('Please fill in all fields with valid values before predicting placement.');
        return;
      }

      predictionForm.classList.add('loading');
      const submitButton = predictionForm.querySelector('button[type="submit"]');
      if (submitButton) {
        submitButton.disabled = true;
      }
    });
  }

  const renderDatasetChart = function () {
    const chartNode = document.getElementById('datasetChart');
    if (!chartNode) {
      return;
    }

    const dataScript = document.getElementById('dashboard-data');
    if (!dataScript) {
      return;
    }

    const dashboard = JSON.parse(dataScript.textContent || '{}');
    const labels = ['Students', 'Placed', 'Avg CGPA'];
    const values = [
      dashboard.student_count ?? 0,
      dashboard.placed_count ?? 0,
      dashboard.average_cgpa ?? 0,
    ];

    new Chart(chartNode, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Dashboard summary',
          data: values,
          backgroundColor: ['#60a5fa', '#22c55e', '#2563eb'],
          borderRadius: 10,
        }],
      },
      options: {
        maintainAspectRatio: false,
        responsive: true,
        plugins: {
          legend: {
            display: false,
          },
        },
        scales: {
          y: {
            beginAtZero: false,
            ticks: {
              precision: 0,
            },
          },
        },
      },
    });
  };

  const renderPerformanceChart = function () {
    const chartNode = document.getElementById('performanceChart');
    if (!chartNode) {
      return;
    }

    const dataScript = document.getElementById('model-performance-data');
    if (!dataScript) {
      return;
    }

    const performance = JSON.parse(dataScript.textContent || '{}');
    const metrics = performance.metrics || [];

    if (!metrics.length) {
      return;
    }

    const labels = metrics.map(function (item) {
      return item.Model || 'Model';
    });

    const datasets = [
      {
        label: 'Accuracy',
        data: metrics.map(function (item) { return Number(item['Accuracy']) || 0; }),
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.2)',
        tension: 0.3,
        fill: false,
      },
      {
        label: 'F1-score',
        data: metrics.map(function (item) { return Number(item['F1-score']) || 0; }),
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
        tension: 0.3,
        fill: false,
      },
    ];

    new Chart(chartNode, {
      type: 'line',
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: {
        maintainAspectRatio: false,
        responsive: true,
        scales: {
          y: {
            beginAtZero: false,
            min: 0,
            max: 1.1,
          },
        },
      },
    });
  };

  renderDatasetChart();
  renderPerformanceChart();
});

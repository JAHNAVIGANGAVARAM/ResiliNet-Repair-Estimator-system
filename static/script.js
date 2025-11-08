const form = document.getElementById('prediction-form');
const submitButton = form.querySelector('[data-role="submit"]');
const resultsCard = document.getElementById('results-card');
const toastEl = document.getElementById('toast');
const timelineBar = document.querySelector('.timeline__bar');
const timelineMarkers = document.querySelectorAll('.timeline__marker');
const demoButton = document.querySelector('[data-action="demo"]');

const setResultText = (key, value) => {
    const node = document.querySelector(`[data-result="${key}"]`);
    if (node) {
        node.textContent = value;
    }
};

const setTimelineSteps = (steps = []) => {
    ['step1', 'step2', 'step3'].forEach((key, index) => {
        setResultText(key, steps[index] || '—');
    });
};

const toggleLoading = (isLoading) => {
    document.body.classList.toggle('is-loading', isLoading);
    submitButton.disabled = isLoading;
};

const showToast = (message, tone = 'error', duration = 4200) => {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.className = `toast show toast--${tone}`;
    setTimeout(() => {
        toastEl.classList.remove('show');
    }, duration);
};

const capitalise = (value = '') => value.charAt(0).toUpperCase() + value.slice(1);

const populateDemo = () => {
    document.getElementById('cause').value = 'Major fiber cut';
    document.getElementById('country').value = 'India';
    document.getElementById('region').value = 'Hyderabad';
    document.getElementById('severity').value = 'high';
};

if (demoButton) {
    demoButton.addEventListener('click', populateDemo);
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const cause = document.getElementById('cause').value.trim();
    if (!cause) {
        showToast('Please provide the cause of the shutdown.', 'error');
        return;
    }

    const formData = new FormData(form);

    try {
        toggleLoading(true);
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.error || 'Failed to fetch predictions.');
        }

        const {
            repair_duration: repairDuration,
            recommended_solution: recommendedSolution,
            severity_label: severityLabel,
            matched_cause: matchedCause,
            reference_count: referenceCount,
            reference_median: referenceMedian,
            timeline_progress: timelineProgress,
            timeline_steps: timelineSteps,
            eta_label: etaLabel
        } = payload;

        setResultText('duration', repairDuration || '—');
        setResultText('solution', recommendedSolution || '—');
        setResultText('severity', capitalise(severityLabel || 'pending'));
        setResultText('matched-cause', matchedCause ? capitalise(matchedCause) : '—');
        setResultText('reference-count', referenceCount ? `${referenceCount}` : '0');
        setResultText('eta', etaLabel || 'TBD');
        setResultText('reference-median', referenceMedian || '—');

        setTimelineSteps(timelineSteps || []);

        if (timelineBar) {
            const width = Math.min(100, Math.max(10, Number(timelineProgress) || 0));
            timelineBar.style.width = `${width}%`;
        }

        timelineMarkers.forEach((marker) => {
            const stage = Number(marker.dataset.stage || 0);
            const progress = Number(timelineProgress) || 0;
            marker.classList.toggle('is-active', progress >= stage);
        });

        resultsCard.classList.add('visible');
        showToast('Prediction updated successfully.', 'success', 3000);
    } catch (error) {
        console.error('Prediction error:', error);
        showToast(error.message || 'An error occurred while fetching predictions.', 'error');
    } finally {
        toggleLoading(false);
    }
});
const form = document.getElementById("generate-form");
const companyInput = document.getElementById("company");
const personInput = document.getElementById("person");
const generateButton = document.getElementById("generate-button");
const statusText = document.getElementById("status-text");
const emptyState = document.getElementById("empty-state");
const loadingState = document.getElementById("loading-state");
const errorState = document.getElementById("error-state");
const resultsContainer = document.getElementById("results");
const progressLog = document.getElementById("progress-log");
let pollTimer = null;

function setLoading(isLoading) {
  generateButton.disabled = isLoading;
  loadingState.classList.toggle("hidden", !isLoading);
  emptyState.classList.toggle("hidden", isLoading);
}

function clearStates() {
  errorState.classList.add("hidden");
  errorState.textContent = "";
  resultsContainer.classList.add("hidden");
  resultsContainer.innerHTML = "";
  progressLog.innerHTML = "";
}

function sourceChips(sources) {
  if (!sources || !sources.length) {
    return `<p class="support-note">No supporting links attached.</p>`;
  }
  return `
    <div class="sources-row">
      ${sources
        .map(
          (source) => `
          <a class="source-chip" href="${source.url}" target="_blank" rel="noreferrer noopener">
            <span>${source.title || source.domain}</span>
          </a>
        `
        )
        .join("")}
    </div>
  `;
}

function renderProfiles(profiles) {
  if (!profiles || !profiles.length) {
    return "";
  }
  return `
    <section class="results-section">
      <h3>Official Links</h3>
      <div class="profiles-row">
        ${profiles
          .map(
            (profile) => `
            <a class="profile-pill" href="${profile.url}" target="_blank" rel="noreferrer noopener">
              <span>${profile.label}</span>
            </a>
          `
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderStrategy(strategyItems) {
  return `
    <section class="results-section">
      <h3>Current Marketing</h3>
      <div class="bullet-list">
        ${strategyItems
          .map(
            (item) => `
            <div class="list-card">
              <p>${item.text}</p>
              ${sourceChips(item.sources)}
            </div>
          `
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderGaps(gapItems) {
  return `
    <section class="results-section">
      <h3>Marketing Gaps</h3>
      <div class="gap-list">
        ${gapItems
          .map(
            (item) => `
            <div class="gap-card">
              <p>${item.text}</p>
              ${sourceChips(item.sources)}
            </div>
          `
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderIdea(idea) {
  return `
    <section class="results-section">
      <h3>Instagram Post Idea</h3>
      <div class="idea-grid">
        <div class="idea-block">
          <strong>Hook</strong>
          <p>${idea.hook}</p>
        </div>
        <div class="idea-block">
          <strong>Concept</strong>
          <p>${idea.concept}</p>
        </div>
        <div class="idea-block">
          <strong>Why It Helps</strong>
          <p>${idea.why_it_closes_the_gap}</p>
        </div>
      </div>
      <div class="chip-row">
        ${sourceChips(idea.sources)}
      </div>
    </section>
  `;
}

function renderEmail(email) {
  const emailText = `Subject: ${email.subject}\n\n${email.body}`;
  return `
    <section class="results-section">
      <h3>Outreach Email</h3>
      <div class="email-box">
        <div class="email-meta"><strong>Subject:</strong> ${email.subject}</div>
        <div class="email-body">${email.body}</div>
        <button class="copy-button" data-copy="${encodeURIComponent(emailText)}">Copy Email</button>
      </div>
    </section>
  `;
}

function renderSources(sourceLibrary) {
  return `
    <section class="results-section">
      <h3>Sources Reviewed</h3>
      <div class="bullet-list">
        ${sourceLibrary
          .map(
            (source) => `
            <div class="list-card">
              <p><strong>${source.title}</strong></p>
              <p class="support-note">${source.snippet || source.domain}</p>
              <div class="sources-row">
                <a class="source-chip" href="${source.url}" target="_blank" rel="noreferrer noopener">Open source</a>
              </div>
            </div>
          `
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderResult(payload) {
  const analysis = payload.analysis;
  resultsContainer.innerHTML = `
    ${renderProfiles(payload.profiles)}
    <section class="results-section">
      <h3>Brand Summary</h3>
      <p>${analysis.brand_summary}</p>
    </section>
    <section class="results-section">
      <h3>Likely Target Audience</h3>
      <p>${analysis.likely_target_audience}</p>
    </section>
    ${renderStrategy(analysis.current_marketing_strategy || [])}
    ${renderGaps(analysis.marketing_gaps || [])}
    ${renderIdea(analysis.instagram_post_idea || {})}
    ${renderEmail(analysis.outreach_email || {})}
    ${renderSources(payload.source_library || [])}
  `;
  resultsContainer.classList.remove("hidden");

  resultsContainer.querySelectorAll(".copy-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const decoded = decodeURIComponent(button.dataset.copy);
      await navigator.clipboard.writeText(decoded);
      button.textContent = "Copied";
      setTimeout(() => {
        button.textContent = "Copy Email";
      }, 1500);
    });
  });
}

function renderProgress(events) {
  progressLog.innerHTML = (events || [])
    .slice(-8)
    .map((event) => `<div class="progress-step">${event}</div>`)
    .join("");
}

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Unable to fetch job status.");
  }

  renderProgress(payload.events);
  statusText.textContent = payload.message || "Working...";

  if (payload.status === "completed") {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    setLoading(false);
    emptyState.classList.add("hidden");
    renderResult(payload.result);
    statusText.textContent = `Completed using the ${payload.result.backend} backend.`;
    return;
  }

  if (payload.status === "failed") {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    setLoading(false);
    statusText.textContent = "Something went wrong.";
    errorState.textContent = payload.error || "Generation failed.";
    errorState.classList.remove("hidden");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const company = companyInput.value.trim();
  const person = personInput.value.trim();

  if (!company) {
    errorState.textContent = "Company name is required.";
    errorState.classList.remove("hidden");
    return;
  }

  clearStates();
  setLoading(true);
  statusText.textContent = "Searching live sources and generating a grounded brief...";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ company, person }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Generation failed.");
    }

    renderProgress(["Queued"]);
    pollTimer = setInterval(() => {
      pollJob(payload.job_id).catch((error) => {
        if (pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        setLoading(false);
        statusText.textContent = "Something went wrong.";
        errorState.textContent = error.message || "Generation failed.";
        errorState.classList.remove("hidden");
      });
    }, 1200);
    await pollJob(payload.job_id);
  } catch (error) {
    setLoading(false);
    statusText.textContent = "Something went wrong.";
    errorState.textContent = error.message || "Generation failed.";
    errorState.classList.remove("hidden");
  }
});

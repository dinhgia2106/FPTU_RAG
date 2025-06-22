/**
 * Chat Interface JavaScript
 * Xử lý giao diện chat và API calls
 */

class ChatInterface {
  constructor() {
    this.messagesContainer = document.getElementById("messages");
    this.messageInput = document.getElementById("messageInput");
    this.sendButton = document.getElementById("sendButton");
    this.subjectsModal = document.getElementById("subjectsModal");
    this.examplesModal = document.getElementById("examplesModal");
    this.loadingIndicator = document.getElementById("loading");

    // Timer variables
    this.responseStartTime = null;
    this.timerInterval = null;

    this.initializeEventListeners();
    this.loadExamples();
    this.updateCharacterCount();
    this.updateSendButtonState();
  }

  initializeEventListeners() {
    // Send button click
    this.sendButton.addEventListener("click", () => this.sendMessage());

    // Enter key press
    this.messageInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Input changes
    this.messageInput.addEventListener("input", () => {
      this.updateCharacterCount();
      this.updateSendButtonState();
    });

    // Modal close buttons
    document.querySelectorAll(".close-modal").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.target.closest(".modal").style.display = "none";
      });
    });

    // Modal background click
    document.querySelectorAll(".modal").forEach((modal) => {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) {
          modal.style.display = "none";
        }
      });
    });

    // Header buttons
    document.getElementById("showSubjects")?.addEventListener("click", () => {
      this.showSubjects();
    });

    document.getElementById("showExamples")?.addEventListener("click", () => {
      this.showExamples();
    });
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message) return;

    // Add user message to chat
    this.addMessage(message, "user");
    this.messageInput.value = "";
    this.updateCharacterCount();
    this.updateSendButtonState();
    this.showLoading(true);
    this.startResponseTimer();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: message }),
      });

      const data = await response.json();
      const responseTime = this.stopResponseTimer();

      if (response.ok) {
        this.addMessage(data.answer, "assistant", responseTime);

        // Show search results if available
        if (data.search_results && data.search_results.length > 0) {
          this.showSearchResults(data.search_results);
        }
      } else {
        this.addMessage(`Lỗi: ${data.error}`, "error");
      }
    } catch (error) {
      console.error("Error:", error);
      this.addMessage(
        "Có lỗi xảy ra khi gửi tin nhắn. Vui lòng thử lại.",
        "error"
      );
    } finally {
      this.showLoading(false);
    }
  }

  addMessage(content, type, responseTime = null) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent =
      type === "user" ? "U" : type === "assistant" ? "AI" : "!";

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";

    const messageText = document.createElement("div");
    messageText.className = "message-text";
    messageText.innerHTML = this.formatMessage(content);

    messageContent.appendChild(messageText);

    // Add response time for assistant messages
    if (type === "assistant" && responseTime) {
      const timeDiv = document.createElement("div");
      timeDiv.className = "response-time";
      timeDiv.textContent = `Phản hồi trong ${responseTime}`;
      messageContent.appendChild(timeDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    this.messagesContainer.appendChild(messageDiv);
    this.scrollToBottom();
  }

  formatMessage(content) {
    // Convert newlines to <br> and format basic markdown
    return content
      .replace(/\n/g, "<br>")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`(.*?)`/g, "<code>$1</code>");
  }

  showSearchResults(results) {
    const resultsDiv = document.createElement("div");
    resultsDiv.className = "search-results";

    const header = document.createElement("div");
    header.className = "search-results-header";
    header.innerHTML = `
      <div class="search-results-title">
        Nguồn tham khảo (${results.length} kết quả)
      </div>
      <div class="search-results-toggle">v</div>
    `;

    const content = document.createElement("div");
    content.className = "search-results-content";

    results.slice(0, 3).forEach((result) => {
      const resultItem = document.createElement("div");
      resultItem.className = "result-item";
      resultItem.innerHTML = `
        <div class="result-subject"><strong>${
          result.subject_code || "N/A"
        }</strong></div>
        <div class="result-content">${this.truncateText(
          result.content,
          100
        )}</div>
      `;
      content.appendChild(resultItem);
    });

    // Add click handler for toggle
    header.addEventListener("click", () => {
      resultsDiv.classList.toggle("expanded");
    });

    resultsDiv.appendChild(header);
    resultsDiv.appendChild(content);
    this.messagesContainer.appendChild(resultsDiv);
    this.scrollToBottom();
  }

  truncateText(text, maxLength) {
    return text.length > maxLength
      ? text.substring(0, maxLength) + "..."
      : text;
  }

  showLoading(show) {
    if (this.loadingIndicator) {
      this.loadingIndicator.style.display = show ? "flex" : "none";
    }
    this.sendButton.disabled = show;
  }

  startResponseTimer() {
    this.responseStartTime = Date.now();
    const timerElement = document.getElementById("responseTimer");

    this.timerInterval = setInterval(() => {
      if (timerElement && this.responseStartTime) {
        const elapsed = Math.floor(
          (Date.now() - this.responseStartTime) / 1000
        );
        timerElement.textContent = `${elapsed}s`;
      }
    }, 100);
  }

  stopResponseTimer() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }

    if (this.responseStartTime) {
      const elapsed = Date.now() - this.responseStartTime;
      this.responseStartTime = null;
      return `${(elapsed / 1000).toFixed(1)}s`;
    }

    return null;
  }

  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  updateCharacterCount() {
    const charCountElement = document.querySelector(".character-count");
    if (charCountElement && this.messageInput) {
      const currentLength = this.messageInput.value.length;
      charCountElement.textContent = `${currentLength}/1000`;
    }
  }

  updateSendButtonState() {
    if (this.sendButton && this.messageInput) {
      const hasText = this.messageInput.value.trim().length > 0;
      this.sendButton.disabled = !hasText;
    }
  }

  async showSubjects() {
    try {
      const response = await fetch("/api/subjects");
      const data = await response.json();

      if (response.ok) {
        const subjectsList = document.getElementById("subjectsList");
        subjectsList.innerHTML = "";

        data.subjects.forEach((subject) => {
          const subjectItem = document.createElement("div");
          subjectItem.className = "subject-item";
          subjectItem.innerHTML = `
                        <div class="subject-code">${subject.code}</div>
                        <div class="subject-name">${subject.name}</div>
                        <div class="subject-credits">${subject.credits} tín chỉ</div>
                    `;
          subjectItem.addEventListener("click", () => {
            this.messageInput.value = `Thông tin về môn ${subject.code}`;
            this.subjectsModal.style.display = "none";
          });
          subjectsList.appendChild(subjectItem);
        });

        this.subjectsModal.style.display = "block";
      }
    } catch (error) {
      console.error("Error loading subjects:", error);
    }
  }

  async showExamples() {
    this.examplesModal.style.display = "block";
  }

  async loadExamples() {
    try {
      const response = await fetch("/api/examples");
      const data = await response.json();

      if (response.ok) {
        const examplesList = document.getElementById("examplesList");
        if (examplesList) {
          examplesList.innerHTML = "";

          data.examples.forEach((example) => {
            const exampleItem = document.createElement("div");
            exampleItem.className = "example-item";
            exampleItem.textContent = example;
            exampleItem.addEventListener("click", () => {
              this.messageInput.value = example;
              this.examplesModal.style.display = "none";
            });
            examplesList.appendChild(exampleItem);
          });
        }
      }
    } catch (error) {
      console.error("Error loading examples:", error);
    }
  }
}

// Global function for example buttons
function sendExample(message) {
  const chatInterface = window.chatInterface;
  if (chatInterface) {
    chatInterface.messageInput.value = message;
    chatInterface.sendMessage();
  }
}

// Initialize chat interface when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.chatInterface = new ChatInterface();
});

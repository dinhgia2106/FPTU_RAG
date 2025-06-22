/**
 * Chat Interface JavaScript
 * Xử lý giao diện chat và API calls
 */

// Modern Chat Interface JavaScript
class ChatInterface {
  constructor() {
    this.chatContainer = document.getElementById("chatContainer");
    this.messageInput = document.getElementById("messageInput");
    this.sendButton = document.getElementById("sendButton");
    this.charCount = document.getElementById("charCount");
    this.typingIndicator = document.getElementById("typingIndicator");

    this.isLoading = false;
    this.responseStartTime = null;

    this.initializeEventListeners();
    this.initializeInputHandlers();
  }

  initializeEventListeners() {
    // Send button click
    this.sendButton.addEventListener("click", () => this.sendMessage());

    // Enter key handling
    this.messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea
    this.messageInput.addEventListener("input", () => {
      this.updateCharCount();
      this.autoResizeTextarea();
    });

    // Initial setup
    this.updateCharCount();
  }

  initializeInputHandlers() {
    this.messageInput.addEventListener("input", () => {
      const hasText = this.messageInput.value.trim().length > 0;
      this.sendButton.disabled = !hasText;

      if (hasText) {
        this.sendButton.classList.remove("disabled:bg-gray-600");
        this.sendButton.classList.add("bg-primary-600", "hover:bg-primary-700");
      } else {
        this.sendButton.classList.add("disabled:bg-gray-600");
        this.sendButton.classList.remove(
          "bg-primary-600",
          "hover:bg-primary-700"
        );
      }
    });
  }

  updateCharCount() {
    const count = this.messageInput.value.length;
    this.charCount.textContent = count;

    if (count > 800) {
      this.charCount.classList.add("text-red-400");
      this.charCount.classList.remove("text-yellow-400");
    } else if (count > 600) {
      this.charCount.classList.add("text-yellow-400");
      this.charCount.classList.remove("text-red-400");
    } else {
      this.charCount.classList.remove("text-red-400", "text-yellow-400");
    }
  }

  autoResizeTextarea() {
    this.messageInput.style.height = "auto";
    this.messageInput.style.height =
      Math.min(this.messageInput.scrollHeight, 120) + "px";
  }

  // Smart multihop detection based on query content
  shouldUseMultihop(query) {
    const queryLower = query.toLowerCase().trim();

    // Explicit multihop triggers
    const multihopTriggers = [
      "và các môn tiên quyết",
      "và môn tiên quyết",
      "cùng với môn tiên quyết",
      "kèm theo môn tiên quyết",
      "và các môn liên quan",
      "thông tin đầy đủ",
      "thông tin chi tiết",
      "chi tiết về",
      "mở rộng thông tin",
      "phân tích chi tiết",
      "tổng quan về",
      "lộ trình học",
      "so sánh với",
      "liên quan đến",
    ];

    // Check for explicit multihop triggers
    for (const trigger of multihopTriggers) {
      if (queryLower.includes(trigger)) {
        return true;
      }
    }

    // Advanced patterns that might need multihop
    const complexPatterns = [
      /(.+)\s+(và|cùng)\s+(.+)/, // "X và Y" patterns
      /so sánh\s+(.+)\s+với\s+(.+)/, // Comparison patterns
      /mối quan hệ\s+giữa\s+(.+)/, // Relationship patterns
      /lộ trình\s+từ\s+(.+)\s+đến\s+(.+)/, // Learning path patterns
    ];

    for (const pattern of complexPatterns) {
      if (pattern.test(queryLower)) {
        return true;
      }
    }

    return false; // Default: single-hop search
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message || this.isLoading) return;

    // Smart multihop detection
    const enableMultihop = this.shouldUseMultihop(message);

    // Add user message to chat (bên phải)
    this.addUserMessage(message);

    // Clear input
    this.messageInput.value = "";
    this.updateCharCount();
    this.autoResizeTextarea();
    this.sendButton.disabled = true;

    // Show inline typing indicator
    this.showTypingIndicator();

    try {
      this.responseStartTime = Date.now();

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          multihop: enableMultihop,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      const responseTime = Date.now() - this.responseStartTime;

      this.hideTypingIndicator();
      this.addAssistantMessage(data.answer, {
        responseTime,
        isQuick: data.metadata?.is_quick_response || false,
        hasFollowup: data.multihop_info?.has_followup || false,
        subjectsCovered: data.metadata?.subjects_covered || 0,
        autoMultihop: enableMultihop,
      });
    } catch (error) {
      this.hideTypingIndicator();
      this.addErrorMessage(`Lỗi: ${error.message}`);
    }
  }

  addUserMessage(message) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "user-message flex justify-end mb-6 w-full";

    messageDiv.innerHTML = `
      <div class="message-content bg-gradient-to-r from-primary-500 to-primary-600 text-white p-4 rounded-2xl rounded-br-md max-w-[70%] shadow-lg">
        ${this.formatMessage(message)}
      </div>
    `;

    this.chatContainer.appendChild(messageDiv);
    this.scrollToBottom();
  }

  addAssistantMessage(message, metadata = {}) {
    const messageDiv = document.createElement("div");
    messageDiv.className =
      "assistant-message flex items-start gap-4 mb-6 w-full";

    const responseTimeClass = metadata.isQuick
      ? "text-green-400"
      : metadata.hasFollowup
      ? "text-yellow-400"
      : "text-gray-400";

    let responseTypeText = "Tìm kiếm thường";
    if (metadata.isQuick) {
      responseTypeText = "Phản hồi nhanh";
    } else if (metadata.hasFollowup) {
      responseTypeText = "Tìm kiếm đa cấp";
    } else if (metadata.autoMultihop) {
      responseTypeText = "Tìm kiếm thông minh";
    }

    messageDiv.innerHTML = `
      <div class="w-10 h-10 bg-gradient-to-r from-primary-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
        <i class="fas fa-robot text-white"></i>
      </div>
      <div class="message-content bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 text-gray-100 p-6 rounded-2xl rounded-tl-md max-w-[80%] shadow-lg">
        ${this.formatMessage(message)}
        <div class="response-time ${responseTimeClass} text-xs mt-3 flex items-center justify-between">
          <span>${responseTypeText}</span>
          <span>${(metadata.responseTime / 1000).toFixed(1)}s</span>
        </div>
      </div>
    `;

    this.chatContainer.appendChild(messageDiv);
    this.scrollToBottom();
  }

  addErrorMessage(message) {
    const messageDiv = document.createElement("div");
    messageDiv.className =
      "assistant-message flex items-start gap-4 mb-6 w-full";

    messageDiv.innerHTML = `
      <div class="w-10 h-10 bg-gradient-to-r from-red-500 to-red-600 rounded-full flex items-center justify-center flex-shrink-0">
        <i class="fas fa-exclamation-triangle text-white"></i>
      </div>
      <div class="bg-red-900/20 border border-red-500/30 text-red-300 p-4 rounded-2xl max-w-[80%] flex items-center gap-3">
        <div>${this.formatMessage(message)}</div>
      </div>
    `;

    this.chatContainer.appendChild(messageDiv);
    this.scrollToBottom();
  }

  formatMessage(message) {
    // Convert markdown-like formatting
    return message
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(
        /`(.*?)`/g,
        '<code class="bg-gray-700/50 px-2 py-1 rounded text-sm">$1</code>'
      )
      .replace(/\n/g, "<br>")
      .replace(
        /#{1,6}\s*(.*?)(?:\n|$)/g,
        '<h3 class="text-lg font-semibold mt-4 mb-2 text-primary-300">$1</h3>'
      )
      .replace(/^\* (.+)$/gm, '<li class="ml-4">$1</li>')
      .replace(
        /(<li.*<\/li>)/s,
        '<ul class="list-disc list-inside space-y-1 my-2">$1</ul>'
      );
  }

  showTypingIndicator() {
    this.isLoading = true;
    this.typingIndicator.classList.remove("hidden");
    this.chatContainer.appendChild(this.typingIndicator);
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    this.isLoading = false;
    this.typingIndicator.classList.add("hidden");
    // Remove from chat container if it's there
    if (this.typingIndicator.parentNode === this.chatContainer) {
      this.chatContainer.removeChild(this.typingIndicator);
    }
  }

  scrollToBottom() {
    setTimeout(() => {
      this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }, 100);
  }
}

// Modal functions
function showSubjects() {
  const modal = document.getElementById("subjectsModal");
  const content = document.getElementById("subjectsContent");

  modal.classList.remove("hidden");
  content.innerHTML =
    '<div class="text-center text-gray-400">Đang tải...</div>';

  fetch("/api/subjects")
    .then((response) => response.json())
    .then((data) => {
      if (data.subjects) {
        content.innerHTML = `
          <div class="grid gap-3">
            ${data.subjects
              .map(
                (subject) => `
              <div class="subject-item" onclick="sendSampleQuery('${subject.code} là môn gì?')">
                <div class="subject-code">${subject.code}</div>
                <div class="subject-name">${subject.name}</div>
                <div class="subject-details">
                  <span>Tín chỉ: ${subject.credits}</span>
                  <span>Kỳ: ${subject.semester}</span>
                </div>
              </div>
            `
              )
              .join("")}
          </div>
        `;
      }
    })
    .catch((error) => {
      content.innerHTML =
        '<div class="error-message">Không thể tải danh sách môn học</div>';
    });
}

function showExamples() {
  fetch("/api/examples")
    .then((response) => response.json())
    .then((data) => {
      if (data.examples) {
        const exampleButtons = data.examples
          .map(
            (example) =>
              `<button onclick="sendSampleQuery('${example}')" class="bg-gray-700/30 hover:bg-gray-600/40 p-3 rounded-lg text-left text-sm transition-all duration-200 border border-gray-600/30 hover:border-gray-500/50 w-full">
            <i class="fas fa-question-circle mr-2 text-primary-400"></i>${example}
          </button>`
          )
          .join("");

        // Show examples in a simple alert or create a modal
        alert("Câu hỏi mẫu:\n" + data.examples.join("\n"));
      }
    });
}

function closeModal(modalId) {
  document.getElementById(modalId).classList.add("hidden");
}

function sendSampleQuery(query) {
  const chatInterface = window.chatInterface;
  if (chatInterface) {
    chatInterface.messageInput.value = query;
    chatInterface.updateCharCount();
    chatInterface.sendMessage();
  }

  // Close any open modals
  closeModal("subjectsModal");
}

// Initialize chat interface when page loads
document.addEventListener("DOMContentLoaded", () => {
  window.chatInterface = new ChatInterface();
});

// Close modals when clicking outside
document.addEventListener("click", (e) => {
  if (
    e.target.classList.contains("fixed") &&
    e.target.classList.contains("inset-0")
  ) {
    e.target.classList.add("hidden");
  }
});

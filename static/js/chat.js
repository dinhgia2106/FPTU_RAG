/**
 * Chat Interface JavaScript
 * Xử lý giao diện chat và API calls
 */

// Modern Chat Interface JavaScript
class ChatInterface {
  constructor() {
    this.chatContainer = document.getElementById("chatContainer");
    this.chatMessages = document.getElementById("chatMessages");
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

    this.chatMessages.appendChild(messageDiv);
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

    this.chatMessages.appendChild(messageDiv);
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

    this.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();
  }

  formatMessage(message) {
    // Xử lý bảng markdown trước
    let formattedMessage = this.processMarkdownTables(message);

    // Xử lý các định dạng markdown khác
    formattedMessage = formattedMessage
      // Headers (# ## ###)
      .replace(
        /^### (.*?)$/gm,
        '<h3 class="text-lg font-semibold mt-4 mb-2 text-primary-300">$1</h3>'
      )
      .replace(
        /^## (.*?)$/gm,
        '<h2 class="text-xl font-bold mt-4 mb-3 text-primary-200">$1</h2>'
      )
      .replace(
        /^# (.*?)$/gm,
        '<h1 class="text-2xl font-bold mt-4 mb-3 text-white">$1</h1>'
      )

      // Bold và Italic
      .replace(
        /\*\*(.*?)\*\*/g,
        '<strong class="font-semibold text-primary-200">$1</strong>'
      )
      .replace(/\*(.*?)\*/g, '<em class="italic text-gray-300">$1</em>')

      // Code inline
      .replace(
        /`(.*?)`/g,
        '<code class="bg-gray-700/70 px-2 py-1 rounded text-sm font-mono text-green-300">$1</code>'
      )

      // Numbered lists (1. 2. 3.)
      .replace(/^\d+\.\s+(.+)$/gm, '<li class="ml-6 mb-1 list-decimal">$1</li>')

      // Bullet points (* -)
      .replace(/^[\*\-]\s+(.+)$/gm, '<li class="ml-6 mb-1 list-disc">$1</li>')

      // Line breaks
      .replace(/\n\n/g, "<br><br>")
      .replace(/\n/g, "<br>");

    // Wrap consecutive list items in ul/ol tags
    formattedMessage = this.wrapListItems(formattedMessage);

    // Xử lý các pattern đặc biệt cho dữ liệu FPTU
    formattedMessage = this.processSpecialPatterns(formattedMessage);

    return formattedMessage;
  }

  processMarkdownTables(text) {
    // Tìm các bảng markdown (có dấu |)
    const tableRegex = /(\|.*\|.*\n)+/g;

    return text.replace(tableRegex, (match) => {
      const rows = match.trim().split("\n");
      if (rows.length < 2) return match;

      // Xử lý header row
      const headerRow = rows[0];
      const headerCells = headerRow
        .split("|")
        .map((cell) => cell.trim())
        .filter((cell) => cell);

      // Skip separator row (thường là dòng thứ 2 với dấu -)
      let dataRows = rows.slice(2);

      // Nếu không có separator, lấy tất cả rows từ dòng 2
      if (dataRows.length === 0 && rows.length > 1) {
        dataRows = rows.slice(1);
      }

      // Fix for handling pipe format lines like "| TMG301 | Scopus Q3, Q4 | 2 điểm | Bài báo được chấp nhận/xuất bản |"
      // by ensuring consistent cell structure
      dataRows = dataRows.map(row => {
        // If row starts with | and ends with |, ensure consistent splitting
        if (row.trim().startsWith('|') && row.trim().endsWith('|')) {
          const cleanRow = row.trim();
          return cleanRow;
        }
        return row;
      });

      let tableHTML = '<div class="overflow-x-auto my-4">';
      tableHTML +=
        '<table class="min-w-full border border-gray-600/50 rounded-lg overflow-hidden">';

      // Header
      if (headerCells.length > 0) {
        tableHTML += '<thead class="bg-gray-700/50">';
        tableHTML += "<tr>";
        headerCells.forEach((cell) => {
          tableHTML += `<th class="px-4 py-3 text-left text-sm font-semibold text-primary-300 border-b border-gray-600/50">${cell}</th>`;
        });
        tableHTML += "</tr>";
        tableHTML += "</thead>";
      }

      // Body
      if (dataRows.length > 0) {
        tableHTML += "<tbody>";
        dataRows.forEach((row, index) => {
          // Normalize row format for lines that may have extra spaces or inconsistent formatting
          let rowContent = row;
          if (typeof rowContent === 'string') {
            // Remove leading and trailing whitespace
            rowContent = rowContent.trim();
            // Remove leading and trailing | if they exist
            if (rowContent.startsWith('|')) {
              rowContent = rowContent.substring(1);
            }
            if (rowContent.endsWith('|')) {
              rowContent = rowContent.substring(0, rowContent.length - 1);
            }
          }
          
          const cells = rowContent
            .split('|')
            .map(cell => cell.trim())
            .filter(cell => cell !== '');
            
          if (cells.length > 0) {
            const rowClass =
              index % 2 === 0 ? "bg-gray-800/30" : "bg-gray-800/50";
            tableHTML += `<tr class="${rowClass} hover:bg-gray-700/30 transition-colors">`;
            cells.forEach((cell) => {
              tableHTML += `<td class="px-4 py-3 text-sm text-gray-200 border-b border-gray-700/50">${cell}</td>`;
            });
            tableHTML += "</tr>";
          }
        });
        tableHTML += "</tbody>";
      }

      tableHTML += "</table>";
      tableHTML += "</div>";

      return tableHTML;
    });
  }

  wrapListItems(text) {
    // Wrap numbered list items
    text = text.replace(
      /(<li class="ml-6 mb-1 list-decimal">.*?<\/li>)(\s*<br>\s*<li class="ml-6 mb-1 list-decimal">.*?<\/li>)*/g,
      (match) => {
        return `<ol class="list-decimal list-inside space-y-1 my-3 ml-4">${match.replace(
          /<br>/g,
          ""
        )}</ol>`;
      }
    );

    // Wrap bullet list items
    text = text.replace(
      /(<li class="ml-6 mb-1 list-disc">.*?<\/li>)(\s*<br>\s*<li class="ml-6 mb-1 list-disc">.*?<\/li>)*/g,
      (match) => {
        return `<ul class="list-disc list-inside space-y-1 my-3 ml-4">${match.replace(
          /<br>/g,
          ""
        )}</ul>`;
      }
    );

    return text;
  }

  processSpecialPatterns(text) {
    // Highlight mã môn học
    text = text.replace(
      /\b([A-Z]{2,4}\d{3}[a-z]*)\b/g,
      '<span class="bg-blue-900/30 text-blue-300 px-2 py-1 rounded font-mono text-sm">$1</span>'
    );

    // Highlight năm học và kỳ
    text = text.replace(
      /\b(Kỳ|Ky|Kì|Ki)\s+(\d+)\b/g,
      '<span class="bg-green-900/30 text-green-300 px-2 py-1 rounded text-sm font-medium">$1 $2</span>'
    );

    // Highlight số tín chỉ
    text = text.replace(
      /(\d+)\s+(tín chỉ|tin chi)/gi,
      '<span class="bg-yellow-900/30 text-yellow-300 px-2 py-1 rounded text-sm font-medium">$1 $2</span>'
    );

    // Highlight từ khóa quan trọng
    const keywords = [
      "Môn tiên quyết",
      "Prerequisites",
      "CLO",
      "Learning Outcomes",
      "Coursera",
      "Deep Learning",
      "Machine Learning",
      "AI",
    ];

    keywords.forEach((keyword) => {
      const regex = new RegExp(`\\b(${keyword})\\b`, "gi");
      text = text.replace(
        regex,
        '<span class="bg-purple-900/30 text-purple-300 px-1 py-0.5 rounded text-sm font-medium">$1</span>'
      );
    });

    return text;
  }

  showTypingIndicator() {
    this.isLoading = true;
    // Tạo wrapper div với layout phù hợp
    const typingWrapper = document.createElement("div");
    typingWrapper.className = "flex items-start space-x-4 mb-6";
    typingWrapper.innerHTML = `
      <div class="w-10 h-10 bg-gradient-to-r from-primary-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
        <i class="fas fa-robot text-white"></i>
      </div>
      <div class="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 p-4 rounded-2xl flex items-center gap-3">
        <div class="typing-dots">
          <div></div>
          <div></div>
          <div></div>
        </div>
        <span class="text-gray-400 text-sm">AI đang suy nghĩ...</span>
      </div>
    `;

    this.typingIndicator.innerHTML = "";
    this.typingIndicator.appendChild(typingWrapper);
    this.typingIndicator.classList.remove("hidden");
    this.chatMessages.appendChild(this.typingIndicator);
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

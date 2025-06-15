document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");
  // const toolButton = document.getElementById('tool-button'); // Sẽ dùng sau
  // const micButton = document.getElementById('mic-button'); // Sẽ dùng sau

  // Đường dẫn tới avatar (bạn cần có file này trong static/images)
  const userAvatarUrl = "{{ url_for('static', filename='user_avatar.png') }}"; // Hoặc để trống nếu không có
  const botAvatarUrl = "{{ url_for('static', filename='bot_avatar.png') }}"; // Thay bằng đường dẫn thực tế

  function addMessageToChatBox(message, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message");

    const avatar = document.createElement("img");
    avatar.classList.add("avatar");

    const messageContentDiv = document.createElement("div");
    messageContentDiv.classList.add("message-content");
    const p = document.createElement("p");
    if (sender === "bot") {
      p.innerHTML = marked.parse(message);
    } else {
      p.innerHTML = message;
    }
    messageContentDiv.appendChild(p);

    if (sender === "user") {
      messageDiv.classList.add("user-message");
      avatar.src = userAvatarUrl;
      avatar.alt = "User";
      // Trong CSS, user avatar đang bị ẩn bằng visibility: hidden.
      // Nếu muốn hiện, bỏ comment dòng dưới và đảm bảo có ảnh.
      // avatar.style.visibility = "visible";
      messageDiv.appendChild(messageContentDiv); // Nội dung trước
      messageDiv.appendChild(avatar); // Avatar sau
    } else {
      // bot
      messageDiv.classList.add("bot-message");
      avatar.src = botAvatarUrl;
      avatar.alt = "Bot";
      messageDiv.appendChild(avatar); // Avatar trước
      messageDiv.appendChild(messageContentDiv); // Nội dung sau
    }

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Tự động cuộn xuống
  }

  async function sendMessage() {
    const messageText = userInput.value.trim();
    if (messageText === "") return;

    addMessageToChatBox(messageText, "user");
    userInput.value = ""; // Xóa input sau khi gửi
    userInput.focus();

    // Hiển thị "Bot đang gõ..." (Tùy chọn)
    // addMessageToChatBox("<em>Bot đang soạn tin...</em>", 'bot');

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: messageText }),
      });

      // Xóa tin nhắn "Bot đang gõ..." nếu có
      // const typingIndicator = chatBox.querySelector(".bot-message em");
      // if(typingIndicator && typingIndicator.textContent.includes("soạn tin")){
      //     typingIndicator.closest('.message').remove();
      // }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Network response was not ok");
      }

      const data = await response.json();
      addMessageToChatBox(data.response, "bot");
    } catch (error) {
      console.error("Error sending message:", error);
      addMessageToChatBox(
        `Rất tiếc, đã có lỗi xảy ra: ${error.message}`,
        "bot"
      );
    }
  }

  sendButton.addEventListener("click", sendMessage);
  userInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      sendMessage();
    }
  });

  // Placeholder cho các nút khác
  // toolButton.addEventListener('click', () => console.log('Tool button clicked'));
  // micButton.addEventListener('click', () => console.log('Mic button clicked'));

  // Tin nhắn chào mừng ban đầu (ví dụ, nếu bạn không muốn nó hardcode trong HTML)
  // addMessageToChatBox("Alo alo! Mình nghe rõ đây — bạn cần mình giúp gì nè? 😊", 'bot');
});

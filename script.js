function addMessage(msg, sender) {
    const box = document.getElementById("chatBox");
    const div = document.createElement("div");
    div.innerText = sender + ": " + msg;
    box.appendChild(div);
}

function sendMessage() {
    const input = document.getElementById("userInput");
    const message = input.value;

    addMessage(message, "You");

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(res => res.json())
    .then(data => {
        console.log("Response:", data);

        if (data.status === 'success') {
            addMessage(data.reply, "Bot");
        } else {
            addMessage(data.message, "Bot");
        }
    })
    .catch(err => {
        console.error(err);
        addMessage("Error connecting to server", "Bot");
    });

    input.value = "";
}
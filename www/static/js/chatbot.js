chatbot_api = "/chatbot";

function send(ele) {
    if(event.key === 'Enter') {
        var msg = ele.value;
        ele.value = "";
        if (msg.length > 0) {
            set_user_msg(msg);
            var params = {q: msg}
            var resp = fetch(chatbot_api, {
                            method: 'POST',
                            headers: {
                                'Accept': 'application/json, text/plain, */*',
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(params)
                        })
                        .then(data => {return data.json();})
                        .then(res  => {
                            console.log(res);
                            set_bot_msg(res["answer"]);
                            window.scrollTo(0,document.body.scrollHeight);
                        });
        }
    }
}


messages_box = document.getElementById('messages');

function set_msg(msg) {
    var element = document.createElement("div");
    element.appendChild(document.createTextNode(msg));
    messages_box.appendChild(element);
    return element;
}

function set_user_msg(msg) {
    var element = set_msg(msg)
    element.className = "container darker";
}

function set_bot_msg(msg) {
    var element = set_msg(msg)
    element.className = "container";
}
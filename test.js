var XMLHttpRequest = require("xhr2");
const HTTP = new XMLHttpRequest();
HTTP.open("GET", "http://127.0.0.1:8000/");
HTTP.send();
HTTP.onreadystatechange = function (e) {
  if (this.readyState == 4 && this.status == 200) {
    console.log(this.responseText);
  }
};

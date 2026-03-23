const bounds = [
    [-39.91001, -23.102037],
    [26.57356, 79.549774]
];

const frames = [
    "outputs/live/regular/pancast_t030.png",
    "outputs/live/regular/pancast_t060.png",
    "outputs/live/regular/pancast_t090.png",
    "outputs/live/regular/pancast_t120.png"
];


let validTimes = [];
let current = 0;
let playing = false;
let timer = null;


const map = L.map("map", {
    zoomControl: false
});

L.control.zoom({
    position: 'bottomright'
}).addTo(map);

map.fitBounds(bounds);

L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 10 }
).addTo(map);

const slider = document.getElementById("slider");
const playpause = document.getElementById("playpause");
const legendToggle = document.getElementById("legendToggle");
const legendBody = document.getElementById("legendBody");
const validTime = document.getElementById("validTime");

slider.max = frames.length - 1;

frames.forEach(function(src) {
    const img = new Image();
    img.src = src;
});

const overlay = L.imageOverlay(frames[current], bounds, {
    opacity: 0.8,
    interactive: false
}).addTo(map);

function updateTime() {
    if (validTimes.length > 0) {
        validTime.innerHTML = validTimes[current];
    }
}

function showFrame(index) {
    current = index;
    slider.value = current;
    overlay.setUrl(frames[current]);
    updateTime();
}

function nextFrame() {
    const next = (current + 1) % frames.length;
    showFrame(next);
}

function stopAnimation() {
    clearInterval(timer);
    timer = null;
    playing = false;
    playpause.innerHTML = "▶";
}

function startAnimation() {
    timer = setInterval(nextFrame, 1200);
    playing = true;
    playpause.innerHTML = "⏸";
}

slider.oninput = function() {
    if (playing) {
        stopAnimation();
    }

    showFrame(parseInt(this.value));
};

playpause.onclick = function() {
    if (playing) {
        stopAnimation();
    } else {
        startAnimation();
    }
};

legendToggle.onclick = function() {
    if (legendBody.style.display === "" || legendBody.style.display === "none") {
        legendBody.style.display = "block";
    } else {
        legendBody.style.display = "none";
    }
};

fetch("latest.json")
.then(response => response.json())
.then(data => {
    validTimes = data.valid_times;
    updateTime();
});
<?php
// ============================================================
// WATER SAFETY AI - USSD Backend
// ============================================================

$sessionId   = $_POST["sessionId"];
$serviceCode = $_POST["serviceCode"];
$phoneNumber = $_POST["phoneNumber"];
$text        = $_POST["text"];

// Database connection
$conn = new mysqli("localhost", "username", "password", "water_safety_ussd");
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$response = "";

if ($text == "") {
    $response = "CON Welcome to Water Safety AI\n";
    $response .= "1. English\n";
    $response .= "2. Chichewa";
} elseif ($text == "1") {
    $response = "CON Water Safety Services\n";
    $response .= "1. Report water quality issue\n";
    $response .= "2. Report device problem\n";
    $response .= "3. Request chlorine refill\n";
    $response .= "4. Leave feedback";
} elseif ($text == "1*1") {
    $stmt = $conn->prepare("INSERT INTO reports (user_id, report_type) SELECT user_id, 'WaterQuality' FROM users WHERE phone_number=?");
    $stmt->bind_param("s", $phoneNumber);
    $stmt->execute();
    $response = "END Water quality issue reported. Thank you.";
} elseif ($text == "1*2") {
    $stmt = $conn->prepare("INSERT INTO reports (user_id, report_type) SELECT user_id, 'DeviceProblem' FROM users WHERE phone_number=?");
    $stmt->bind_param("s", $phoneNumber);
    $stmt->execute();
    $response = "END Device problem reported.";
} elseif ($text == "1*3") {
    $stmt = $conn->prepare("INSERT INTO refills (user_id) SELECT user_id FROM users WHERE phone_number=?");
    $stmt->bind_param("s", $phoneNumber);
    $stmt->execute();
    $response = "END Refill request submitted.";
} elseif ($text == "1*4") {
    $stmt = $conn->prepare("INSERT INTO feedback (user_id, message) SELECT user_id, 'General feedback submitted' FROM users WHERE phone_number=?");
    $stmt->bind_param("s", $phoneNumber);
    $stmt->execute();
    $response = "END Feedback recorded. Thank you.";
} else {
    $response = "END Invalid option. Dial again.";
}

header('Content-type: text/plain');
echo $response;
$conn->close();
?>

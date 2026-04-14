<?php
// ============================================================
// WATER SAFETY AI - USSD Backend (Production Ready)
// ============================================================

require_once("config.php");

// Get POST data safely
$sessionId   = $_POST["sessionId"] ?? '';
$serviceCode = $_POST["serviceCode"] ?? '';
$phoneNumber = $_POST["phoneNumber"] ?? '';
$text        = $_POST["text"] ?? '';

// Normalize input
$textArray = explode("*", $text);
$level = count($textArray);

// Ensure user exists
$stmt = $conn->prepare("INSERT IGNORE INTO users (phone_number) VALUES (?)");
$stmt->bind_param("s", $phoneNumber);
$stmt->execute();

// Get user_id
$stmt = $conn->prepare("SELECT id FROM users WHERE phone_number=?");
$stmt->bind_param("s", $phoneNumber);
$stmt->execute();
$result = $stmt->get_result();
$user = $result->fetch_assoc();
$user_id = $user['id'];

$response = "";

// ============================================================
// USSD FLOW
// ============================================================

if ($text == "") {
    $response = "CON Welcome to Water Safety AI\n";
    $response .= "1. English\n";
    $response .= "2. Chichewa";
}

// ---------------- LANGUAGE SELECTED ----------------
elseif ($text == "1") {
    $response = "CON Water Safety Services\n";
    $response .= "1. Report water quality issue\n";
    $response .= "2. Report device problem\n";
    $response .= "3. Request chlorine refill\n";
    $response .= "4. Leave feedback";
}

// ---------------- WATER QUALITY ----------------
elseif ($text == "1*1") {
    $stmt = $conn->prepare("INSERT INTO reports (user_id, report_type) VALUES (?, 'WaterQuality')");
    $stmt->bind_param("i", $user_id);
    $stmt->execute();

    $response = "END Water quality issue reported. Thank you.";
}

// ---------------- DEVICE PROBLEM ----------------
elseif ($text == "1*2") {
    $stmt = $conn->prepare("INSERT INTO reports (user_id, report_type) VALUES (?, 'DeviceProblem')");
    $stmt->bind_param("i", $user_id);
    $stmt->execute();

    $response = "END Device problem reported.";
}

// ---------------- REFILL REQUEST ----------------
elseif ($text == "1*3") {
    $stmt = $conn->prepare("INSERT INTO refills (user_id) VALUES (?)");
    $stmt->bind_param("i", $user_id);
    $stmt->execute();

    $response = "END Refill request submitted.";
}

// ---------------- FEEDBACK (MULTI-STEP INPUT) ----------------
elseif ($text == "1*4") {
    $response = "CON Please type your feedback:";
}

elseif ($level == 3 && $textArray[1] == "4") {
    $feedback = $textArray[2];

    $stmt = $conn->prepare("INSERT INTO feedback (user_id, message) VALUES (?, ?)");
    $stmt->bind_param("is", $user_id, $feedback);
    $stmt->execute();

    $response = "END Thank you for your feedback.";
}

// ---------------- INVALID ----------------
else {
    $response = "END Invalid option. Try again.";
}

// ============================================================
// RESPONSE
// ============================================================

header('Content-type: text/plain');
echo $response;

// Log session (optional but useful for AI training)
$stmt = $conn->prepare("INSERT INTO ussd_logs (session_id, phone_number, text) VALUES (?, ?, ?)");
$stmt->bind_param("sss", $sessionId, $phoneNumber, $text);
$stmt->execute();

$conn->close();
?>

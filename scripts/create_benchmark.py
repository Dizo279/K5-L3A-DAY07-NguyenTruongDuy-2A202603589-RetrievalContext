import json

with open('d:/AI_ThucChien/Day6/K5-L3A-DAY06-2A202603589-DataIngestion/starter/benchmark/questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

# Dev additions (unanswerable)
unans_dev_1 = {
    'id': 'q_unans_dev_1',
    'split': 'dev',
    'type': 'unanswerable',
    'question': 'What was the specific impact of the Chilean copper mine labor strikes on global refined copper production in 2026 according to the report?',
    'expected_answer': 'Insufficient evidence: The report discusses base metals and copper price trends but does not provide details or figures on Chilean copper mine labor strikes in 2026.',
    'gold_source_id': None,
    'gold_page': None,
    'evidence_quote': None,
    'filter': {}
}

unans_dev_2 = {
    'id': 'q_unans_dev_2',
    'split': 'dev',
    'type': 'unanswerable',
    'question': 'What target price for global uranium concentrate was projected for 2027 in the report?',
    'expected_answer': 'Insufficient evidence: The report does not cover uranium concentrate or nuclear fuel price forecasts.',
    'gold_source_id': None,
    'gold_page': None,
    'evidence_quote': None,
    'filter': {}
}

# Test additions (unanswerable)
unans_test_1 = {
    'id': 'q_unans_test_1',
    'split': 'test',
    'type': 'unanswerable',
    'question': "By what exact percentage was Brazil's soybean harvest projected to contract due to regional drought in 2026?",
    'expected_answer': "Insufficient evidence: The report does not contain data on Brazil's soybean harvest contraction percentage from drought in 2026.",
    'gold_source_id': None,
    'gold_page': None,
    'evidence_quote': None,
    'filter': {}
}

unans_test_2 = {
    'id': 'q_unans_test_2',
    'split': 'test',
    'type': 'unanswerable',
    'question': 'What was the revised official OPEC crude oil output quota for member nations announced in August 2026?',
    'expected_answer': 'Insufficient evidence: The report does not detail specific August 2026 official OPEC crude oil production quotas for member nations.',
    'gold_source_id': None,
    'gold_page': None,
    'evidence_quote': None,
    'filter': {}
}

all_questions = questions + [unans_dev_1, unans_dev_2, unans_test_1, unans_test_2]
print(f'Total questions: {len(all_questions)}')
dev_q = [q for q in all_questions if q['split'] == 'dev']
test_q = [q for q in all_questions if q['split'] == 'test']
print(f'Dev questions: {len(dev_q)} (answerable: {sum(1 for q in dev_q if q["type"] != "unanswerable")}, unanswerable: {sum(1 for q in dev_q if q["type"] == "unanswerable")})')
print(f'Test questions: {len(test_q)} (answerable: {sum(1 for q in test_q if q["type"] != "unanswerable")}, unanswerable: {sum(1 for q in test_q if q["type"] == "unanswerable")})')

with open('data/benchmark_questions_24.json', 'w', encoding='utf-8') as f:
    json.dump(all_questions, f, ensure_ascii=False, indent=2)
print('Successfully saved data/benchmark_questions_24.json')

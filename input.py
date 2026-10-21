import re
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

class TextNormalizer:
    """
    한국어 텍스트 정규화 클래스
    - 종결어미 정규화 (용->요, 삼->요)
    - 반복 표현 정리 (안녕안녕 -> 안녕)
    - 이모티콘/특수문자 정리
    - 의문문/평서문 구분
    """
    
    def __init__(self):
        # 종결어미 정규화 규칙
        self.ending_patterns = [
            (r'([가-힣]+)용([~!?\s]|$)', r'\1요\2'),  # ~용 -> ~요
            (r'([가-힣]+)유([~!?\sㅠㅜ]|$)', r'\1요\2'),  # ~유 -> ~요
            (r'([가-힣]+)삼([~!?\s]|$)', r'\1요\2'),  # ~삼 -> ~요
            (r'([가-힣]+)셈([~!?\s]|$)', r'\1요\2'),  # ~셈 -> ~요
            (r'([가-힣]+)욤([~!?\s]|$)', r'\1요\2'),  # ~욤 -> ~요
            (r'([가-힣]+)당([~!?\s]|$)', r'\1다\2'),  # ~당 -> ~다 (반갑습니당 -> 반갑습니다)
            (r'([가-힣]+)넹([~!?\s]|$)', r'\1네\2'),  # ~넹 -> ~네 (화나넹 -> 화나네)
            (r'([가-힣]+)슴다([~!?\s]|$)', r'\1습니다\2'),  # ~슴다 -> ~습니다
            (r'([가-힣]+)음([~!?\s]|$)', r'\1요\2'),  # ~음 -> ~요
        ]
        
        # 자주 쓰는 오타/발음 변형 사전
        self.typo_dict = {
            # 응 변형
            '웅': '응',
            '웅웅': '응응',
            '엉': '응',
            
            # 미안해 변형
            '미아내': '미안해',
            '미안데': '미안해',
            '미안혜': '미안해',
            '미아네': '미안해',
            '미아내요': '미안해요',
            '미안데요': '미안해요',
            
            # 반가워 변형
            '방가워': '반가워',
            '방가워요': '반가워요',
            '방가웡': '반가워',
            '반가버': '반가워',
            
            # 고마워 변형
            '고마버': '고마워',
            '고마웡': '고마워',
            '고마엉': '고마워',
            '고마버요': '고마워요',
            
            # 그래 변형
            '그래애': '그래',
            '그랭': '그래',
            
            # 했어 변형
            '했써': '했어',
            '했쒀': '했어',
            '해써': '했어',
            '햇어': '했어',
            '해쒀': '했어',
            
            # 지냈어 변형
            '지내써': '지냈어',
            '지냇어': '지냈어',
            '지내쒀': '지냈어',
            
            # 먹었어 변형
            '먹었써': '먹었어',
            '먹엇어': '먹었어',
            '먹어써': '먹었어',
            
            # 봤어 변형
            '봤써': '봤어',
            '봣어': '봤어',
            '바써': '봤어',
            
            # 갔어 변형
            '갔써': '갔어',
            '갓어': '갔어',
            '가써': '갔어',
            
            # 왔어 변형
            '왔써': '왔어',
            '왓어': '왔어',
            '와써': '왔어',
            
            # 잘못 변형
            '잘못해써': '잘못했어',
            '잘못햇어': '잘못했어',
            '잘못해쒀': '잘못했어',
            
            # 하세요 변형
            '하세여': '하세요',
            '하셰요': '하세요',
            '하세염': '하세요',
            
            # 있어 변형
            '이써': '있어',
            '이쒀': '있어',
            '잇어': '있어',
            '이써요': '있어요',
            '잇어요': '있어요',
            
            # 없어 변형
            '업써': '없어',
            '업어': '없어',
            '엄써': '없어',
            '업쒀': '없어',
            '업써요': '없어요',
            '업어요': '없어요',
            
            # 뭐 변형
            '머': '뭐',
            '모': '뭐',
            
            # 그냥 변형
            '그냥': '그냥',
            '그낭': '그냥',
            
            # 진짜 변형
            '진쨔': '진짜',
            '진차': '진짜',
            '진짜아': '진짜',
            '진ㅉ': '진짜',
            
            # 좋아 변형
            '조아': '좋아',
            '조와': '좋아',
            '좋와': '좋아',
            '조항': '좋아',
            '조아요': '좋아요',
            
            # 싫어 변형
            '시러': '싫어',
            '실어': '싫어',
            '싫엉': '싫어',
            '시럼': '싫어',
            '시러요': '싫어요',
            
            # 모르겠어 변형
            '모르겠써': '모르겠어',
            '모르게써': '모르겠어',
            '모르겟어': '모르겠어',
            
            # 알겠어 변형
            '알겠써': '알겠어',
            '알게써': '알겠어',
            '알겟어': '알겠어',
            
            # 그치 변형
            '그치': '그렇지',
            '그쵸': '그렇죠',
            '그죠': '그렇죠',
            
            # 맞아 변형
            '마자': '맞아',
            '맞쟈': '맞아',
            '마쟈': '맞아',
            
            # 아니 변형
            '아니': '아니',
            '아님': '아니',
            '아닌': '아니',
            
            # 되게 변형
            '되게': '되게',
            '디게': '되게',
            '뎌게': '되게',
            
            # 완전 변형
            '완전': '완전',
            '왼전': '완전',
            
            # 대박 변형
            '대박': '대박',
            '댓박': '대박',
        }
        
        # 신조어 → 표준어 사전 (어근 기반)
        self.slang_dict = {
            # 감정 표현 - 킹받
            '킹받아': '화나',
            '킹받네': '짜증나네',
            '킹받는': '화나는',
            '킹받음': '화남',
            '킹받': '화나',
            
            # 빡치다
            '빡쳐': '화나',
            '빡치': '화나',
            '빡친': '화난',
            
            # 재미 표현
            '꿀잼': '재미있어',
            '꿀잼이': '재미있',
            '노잼': '재미없어',
            '노잼이': '재미없',
            '핵노잼': '정말재미없어',
            '잼': '재밌',
            '잼있': '재밌',
            
            # 혐오 표현
            '극혐': '정말싫어',
            '극혐이': '정말싫',
            
            # 힘듦 표현
            '빡세': '매우힘들',
            '빡센': '매우힘든',
            
            # 무시 표현
            '씹혔': '무시당했',
            '씹히': '무시당하',
            '씹어': '무시해',
            
            # 강조 표현
            '레알': '진짜',
            '리얼': '진짜',
            '개쩔': '정말좋',
            '졸라': '정말',
            '존나': '정말',
            
            # 동의/부정
            '인정': '맞아',
            '팩트': '사실',
            '실화': '사실',
            
            # 기타
            '낄끼빠빠': '분위기파악',
            '쩐다': '멋있다',
            '쩔어': '대단해',
            '쩔다': '대단하다',
            '쩌네': '멋있다',
            '쩌러': '멋있어',
            '지린다': '멋있다',
            '지리네': '멋있다',
            '머선129': '무슨일',
            '머선일': '무슨일',
            
            # 인사 표현
            '하이': '안녕하세요',
            '안뇽': '안녕하세요',
            '안냥': '안녕하세요',
            '방가방가': '안녕',
            '하위': '안녕',
            '해위': '안녕',
            '하이요': '안녕',
            '하이염': '안녕',
            '바이요': '안녕히계세요',
            '바이염': '안녕히계세요',
            
            # 감정/상태 표현
            '사롱해': '사랑해',
            '사룽해': '사랑해',
            '뷁': '기분 나빠',
            '쌰갈': '기분나빠',
            '갑분싸': '싸늘하다',
            '현웃': '웃음',
            '뿌엥': '슬퍼',
            '감덩': '감동',
            '갬동': '감동',
            '갬덩': '감동',
            '웃프다': '웃긴데 슬프다',
            
            # 긍정/부정 표현
            '쵝오': '최고',
            '따봉': '최고',
            '야미': '맛있다',
            '붐업': '좋다',
            '붐따': '싫어',
            '이거지': '좋다',
            '이거지!': '좋다',
            
            # 어쩔티비 관련
            '어쩔': '어쩌라고',
            '어쩔티비': '어쩌라고',
            '저쩔티비': '어쩌라고',
            
            # ok 관련
            '오케이': 'ok',
            '오키': 'ok',
            'ㅇㅋ': 'ok',
            
            # 기타 줄임말/신조어
            '알잘딱깔센': '알아서 잘해봐',
            '잼민이': '어린이',
            '팀플': '팀워크',
            '할말하않': '길게 얘기 안할께',
            'gg': '포기',
            '초딩': '초등학생',
            '밤티': '못생겼다',
            '그러를 그러세요': '니 맘대로 하세요',
            '웃겨서 도티 낳음': '개웃기다',
            'ㅇㄱㅈㅉㅇㅇ': '이거진짜에요',
            '뭔말알': '뭔 말인지 알지',
            '가면가': '가면 가',
            'gmg': '가면 가',
            '하면해': '하면 해',
            'hmh': '하면 해',
            '중티': '중국인같다',
            '그니까': '그러니까',
            '나 이거 봤어': '나 이거 봐봤어',
            '왜 이렇게 웃기냐': '웃기다',
            '아자스': '감사합니다',
            '골반통신': '너무 재밌다',
            '밈': '명장면 명대사',
            '짤': '명장면 명대사',
            '가즈아': '가자',
            '레게노': '레전드',
            '졸귀': '너무 귀여워',
            '카공': '카페 공부',
            '어그로': '관심',
            '화이팅': '파이팅',
            '빠이팅': '파이팅',
            'ㄱㅇㅇ': '귀엽다',
            '찐이다': '진짜다',
            '이게 나야': '나 대단하지',
            '꼬봉': '부하',
            '따까리': '부하',
            '취준': '취업준비',
            
            # 줄임말 (초성)
            'ㄹㅇ': '진짜',
            'ㅈㅂ': '제발',
            'ㄱㅅ': '감사',
            'ㅂㅂ': '안녕',
            'ㅅㄱ': '수고해',
            'ㅁㅊ': '미쳤다',
            'ㅊㅋ': '축하해',
            'ㅈㄴ': '엄청',
            'ㅎㅇ': '안녕',
            'ㅇㅈ': '인정',
            'ㅇㄱㄹㅇ': '이거 진짜',
        }
        
        # 강조 접두어 (개-, 핵- 등)
        self.intensifier_prefixes = {
            '개': '정말',
            '핵': '정말',
            '극': '정말',
            '초': '정말',
            '완전': '정말',
        }
        
        # 의문사 패턴
        self.question_words = [
            '뭐', '무엇', '언제', '어디', '누구', '왜', '어떻게', '어느', '몇',
            '얼마', '어떤', '무슨'
        ]
        
        # 의문형 종결어미 패턴
        self.question_endings = [
            r'[가-힣]+니까$',       # ~니까
            r'[가-힣]+ㄹ까$',       # ~ㄹ까
            r'[가-힣]+을까$',       # ~을까
            r'[가-힣]+를까$',       # ~를까
            r'[가-힣]+[^나]니$',    # ~니 (하니, 가니 등, 하지만 ~나니는 제외)
            r'[가-힣]+냐$',         # ~냐
            r'[가-힣][ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ]나$',  # 받침+나 (했나, 갔나 등)
            r'[가-힣]+죠$',         # ~죠 (지요의 줄임말)
            r'[가-힣]+지$',         # ~지 (그렇지, 맞지 등)
        ]
        
    def normalize_endings(self, text):
        """종결어미 정규화"""
        for pattern, replacement in self.ending_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def fix_typos_dict(self, text):
        """사전 기반 오타 교정"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            # 특수문자 분리
            clean_word = re.sub(r'[~!?\s]+$', '', word)
            suffix = word[len(clean_word):]
            
            # 사전에 있으면 교정
            if clean_word in self.typo_dict:
                corrected_words.append(self.typo_dict[clean_word] + suffix)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def convert_slang(self, text):
        """
        신조어 → 표준어 변환
        어근 기반으로 매칭해서 종결어미까지 처리
        예: 킹받넹 -> 화나네, 개좋아 -> 정말좋아, 개빡세 -> 정말매우힘들어
        """
        words = text.split()
        converted_words = []
        
        for word in words:
            # 특수문자 분리
            clean_word = re.sub(r'[~!?\s]+$', '', word)
            suffix = word[len(clean_word):]
            
            converted = False
            
            # 1. 먼저 신조어 사전에서 완전 매칭 시도 (긴 것부터)
            for slang, standard in sorted(self.slang_dict.items(), key=lambda x: len(x[0]), reverse=True):
                if clean_word.startswith(slang):
                    # 어근 뒤의 종결어미 추출
                    ending = clean_word[len(slang):]
                    
                    # 표준어 + 종결어미 결합
                    converted_words.append(standard + ending + suffix)
                    converted = True
                    break
            
            if converted:
                continue
            
            # 2. 신조어 사전에 없으면 강조 접두어 처리 (개좋아 -> 정말좋아)
            for prefix, replacement in self.intensifier_prefixes.items():
                if clean_word.startswith(prefix) and len(clean_word) > len(prefix):
                    rest = clean_word[len(prefix):]
                    # 뒤에 한글이 있을 때만 변환
                    if rest and re.match(r'^[가-힣]', rest):
                        # 접두어 뒤의 단어도 신조어 변환 시도
                        rest_converted = rest
                        for slang, standard in sorted(self.slang_dict.items(), key=lambda x: len(x[0]), reverse=True):
                            if rest.startswith(slang):
                                ending = rest[len(slang):]
                                rest_converted = standard + ending
                                break
                        
                        # 띄어쓰기 추가
                        converted_words.append(replacement + ' ' + rest_converted + suffix)
                        converted = True
                        break
            
            if not converted:
                converted_words.append(word)
        
        return ' '.join(converted_words)
    
    def fix_typos_similarity(self, text, threshold=0.8):
        """
        유사도 기반 오타 자동 교정
        사전의 올바른 단어(values)와 비교하여 유사한 단어 교정
        
        Args:
            text: 교정할 텍스트
            threshold: 유사도 임계값 (0.0~1.0, 기본 0.8)
        """
        if not LEVENSHTEIN_AVAILABLE:
            return text
        
        words = text.split()
        corrected_words = []
        
        # 사전의 올바른 형태들을 기준 단어로 사용
        reference_words = set(self.typo_dict.values())
        
        for word in words:
            # 특수문자 분리
            clean_word = re.sub(r'[~!?\s]+$', '', word)
            suffix = word[len(clean_word):]
            
            # 너무 짧은 단어는 스킵 (1글자는 오탐 가능성 높음)
            if len(clean_word) <= 1:
                corrected_words.append(word)
                continue
            
            # 이미 사전에 있으면 스킵 (이미 처리됨)
            if clean_word in self.typo_dict:
                corrected_words.append(word)
                continue
            
            # 유사도 계산
            best_match = None
            best_ratio = 0
            
            for ref_word in reference_words:
                # 길이 차이가 너무 크면 스킵 (효율성)
                if abs(len(clean_word) - len(ref_word)) > 2:
                    continue
                
                ratio = Levenshtein.ratio(clean_word, ref_word)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = ref_word
            
            # 임계값 이상이면 교정
            if best_ratio >= threshold and best_match:
                corrected_words.append(best_match + suffix)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def remove_repetition(self, text):
        """반복 표현 제거 (안녕안녕 -> 안녕)"""
        # 2음절 이상 단어의 연속 반복 제거
        text = re.sub(r'([가-힣]{2,})\1+', r'\1', text)
        return text
    
    def normalize_elongation(self, text):
        """글자 늘임 정규화 (안녕하세요오오오 -> 안녕하세요)"""
        # 같은 글자가 3번 이상 반복되면 1번으로
        text = re.sub(r'([가-힣ㅋㅎ])\1{2,}', r'\1', text)
        return text
    
    def normalize_emoticons(self, text):
        """이모티콘/특수문자 정리"""
        # 자음/모음 이모티콘 제거 (ㅠ, ㅜ, ㅡ 등) - 먼저 처리
        text = re.sub(r'[ㅠㅜㅡㅗㅓㅏㅣ]+', '', text)
        
        # ㅋㅋ, ㅎㅎ 등은 유지하되 과도한 반복만 제거 (4개 이상 -> 2개)
        text = re.sub(r'ㅋ{4,}', 'ㅋㅋ', text)
        text = re.sub(r'ㅎ{4,}', 'ㅎㅎ', text)
        
        # 과도한 물결표 정리
        text = re.sub(r'~{3,}', '~', text)
        
        # 과도한 느낌표/물음표 정리
        text = re.sub(r'!{3,}', '!', text)
        text = re.sub(r'\?{3,}', '?', text)
        
        return text
    
    def normalize_spacing(self, text):
        """공백 정규화"""
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text
    
    def is_question(self, text):
        """
        의문문 여부 판단
        물음표뿐만 아니라 의문사와 종결어미 패턴도 고려
        
        Returns:
            dict: {
                'is_question': bool,
                'confidence': str ('high', 'medium', 'low'),
                'reason': str (판단 근거)
            }
        """
        # 특수문자 제거한 순수 텍스트
        clean_text = re.sub(r'[~!?\s]+$', '', text).strip()
        
        # 1. 의문사 체크
        has_question_word = any(word in clean_text for word in self.question_words)
        
        # 2. 의문형 종결어미 체크
        has_question_ending = any(re.search(pattern, clean_text) for pattern in self.question_endings)
        
        # 3. 물음표 체크
        has_question_mark = '?' in text
        
        # 판단 로직
        if has_question_word and has_question_ending:
            return {
                'is_question': True,
                'confidence': 'high',
                'reason': '의문사 + 의문형 종결어미'
            }
        elif has_question_word:
            return {
                'is_question': True,
                'confidence': 'high',
                'reason': '의문사 포함'
            }
        elif has_question_ending:
            return {
                'is_question': True,
                'confidence': 'medium',
                'reason': '의문형 종결어미'
            }
        elif has_question_mark:
            return {
                'is_question': True,
                'confidence': 'low',
                'reason': '물음표만 존재 (반말/인사 가능성)'
            }
        else:
            return {
                'is_question': False,
                'confidence': 'high',
                'reason': '평서문'
            }
    
    def normalize(self, text, classify_intent=False, use_similarity=True, similarity_threshold=0.8, convert_slang=True):
        """
        전체 정규화 파이프라인
        
        Args:
            text: 정규화할 텍스트
            classify_intent: True일 경우 의문문 분류 결과도 함께 반환
            use_similarity: 유사도 기반 자동 교정 사용 여부
            similarity_threshold: 유사도 임계값 (0.0~1.0)
            convert_slang: 신조어→표준어 변환 사용 여부
            
        Returns:
            classify_intent=False: 정규화된 텍스트
            classify_intent=True: (정규화된 텍스트, 의문문 분류 결과)
        """
        # 1. 이모티콘/특수문자 정리 (먼저 해서 단어 구분 명확하게)
        text = self.normalize_emoticons(text)
        
        # 2. 신조어 → 표준어 변환 (옵션)
        if convert_slang:
            text = self.convert_slang(text)
        
        # 3. 사전 기반 오타 교정
        text = self.fix_typos_dict(text)
        
        # 4. 유사도 기반 오타 교정 (옵션)
        if use_similarity:
            text = self.fix_typos_similarity(text, threshold=similarity_threshold)
        
        # 5. 종결어미 정규화
        text = self.normalize_endings(text)
        
        # 6. 글자 늘임 정규화
        text = self.normalize_elongation(text)
        
        # 7. 반복 표현 제거
        text = self.remove_repetition(text)
        
        # 8. 공백 정규화
        text = self.normalize_spacing(text)
        
        # 의문문 분류
        intent_result = self.is_question(text)
        
        # 9. 의문문인데 물음표가 없으면 추가 (low 신뢰도 제외)
        if intent_result['is_question'] and intent_result['confidence'] != 'low':
            if not text.endswith('?'):
                # 끝에 특수문자가 있으면 그 앞에 물음표 추가
                if text and text[-1] in '~!':
                    text = text[:-1] + '?' + text[-1]
                else:
                    text = text + '?'
        
        # 의문문 분류 여부에 따라 반환
        if classify_intent:
            return text, intent_result
        else:
            return text


# 사용 예시
if __name__ == "__main__":
    import sys
    
    normalizer = TextNormalizer()
    
    # 명령줄 인자로 텍스트를 받은 경우
    if len(sys.argv) > 1:
        input_text = ' '.join(sys.argv[1:])
        normalized, intent = normalizer.normalize(input_text, classify_intent=True)
        
        print("=" * 60)
        question_mark = "❓" if intent['is_question'] else "💬"
        print(f"{question_mark} 원본: {input_text}")
        print(f"   정규화: {normalized}")
        print(f"   판단: {'의문문' if intent['is_question'] else '평서문'}")
        print(f"   신뢰도: {intent['confidence']}")
        print(f"   근거: {intent['reason']}")
        print("=" * 60)
    
    # 대화형 모드
    else:
        print("=" * 60)
        print("텍스트 정규화 도구")
        print("=" * 60)
        print("사용법:")
        print("  1. 대화형: 그냥 실행 후 텍스트 입력")
        print("  2. 명령줄: python text_normalizer.py '안녕하세용~'")
        print("  3. 종료: 'q' 또는 'quit' 입력")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("텍스트 입력 > ").strip()
                
                if user_input.lower() in ['q', 'quit', '종료']:
                    print("프로그램을 종료합니다.")
                    break
                
                if not user_input:
                    continue
                
                # 정규화 + 의문문 판단
                normalized, intent = normalizer.normalize(user_input, classify_intent=True)
                
                question_mark = "❓" if intent['is_question'] else "💬"
                print(f"\n{question_mark} 원본: {user_input}")
                print(f"   정규화: {normalized}")
                print(f"   판단: {'의문문' if intent['is_question'] else '평서문'} "
                      f"(신뢰도: {intent['confidence']}, 근거: {intent['reason']})")
                print()
                
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
                continue
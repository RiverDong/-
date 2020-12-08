import re


class Blurb:
    def __init__(self, type, value, relativeUrl, desktopRelativeUrl):
        self.type = type
        self.value = value
        self.relativeUrl = relativeUrl
        self.desktopRelativeUrl = desktopRelativeUrl
        self.buttons = []

    def addButton(self, x):
        self.buttons.append(vars(x))


class Button:
    def __init__(self, text, type, value):
        self.text = text
        self.type = type
        self.value = value


### devide by peroid and merge
def split_url_ans(text, pattern = r'\]\([^(]+\)'):
#     pattern = r'\((?:https:\/\/)?www.[^(]+\)'
    url_list = [a[2:][:-1] for a in re.findall(pattern, text)]
    ans_list = []
    ans_raw = re.split(pattern, text)
    for i, a in enumerate(ans_raw):
        if i == len(ans_raw) - 1:
            ans_list.append(a.strip(','))
        else:
            ans_list.append((a + ']').strip(',').strip())
    return ans_list, url_list


def get_answer_list(answers):
    alist = answers.strip().split('. ')
    out_alist=[]
    for a in alist:
        answer = a.strip()
        ans_list, url_list = split_url_ans(answer)
        if len(url_list) > 0:
            for i, u in enumerate(url_list):
                a = re.sub(r'\[(.*)\]', r'<a>\1</a>', ans_list[i])
                if i == len(url_list)-1:
                    ans = (a+ans_list[i+1]).strip()
                    highlights = re.sub(r'\<a\>(.*)\<\/a\>', r'\1', ans)
                    if len(highlights) > 0:
                        out_alist.append([ans,u])
                else:
                    out_alist.append([a.strip(),u])
        else:
            out_alist.append([answer,''])
    return out_alist


def reformat_answer(text,max_ans_len = 150):
    answer_list = get_answer_list(text)
    i=0
    j=1
    ans = answer_list[0][0]
    url = answer_list[0][1]
    while j < len(answer_list):
        next_ans = answer_list[j][0]
        next_url = answer_list[j][1]
        url_check = 1 if len(url) > 0 and len(next_url) > 0 else 0
        if len(ans) + len(next_ans) < max_ans_len and url_check == 0:
            ans += '. ' + next_ans
            url += next_url
            if j == len(answer_list)-1:
                answer_list[i] = [ans,url]
        else:
            answer_list[i] = [ans,url]
            ans = answer_list[j][0]
            url = answer_list[j][1]
            if j == len(answer_list)-1:
                answer_list[i+1] = [ans,url]
            i += 1
        j+=1

    formatted_answer = answer_list[:i+1]
    final_answer = [vars(Blurb('ANSWER',blurb[0],blurb[1],blurb[1])) for blurb in formatted_answer]
    return final_answer
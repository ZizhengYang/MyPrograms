# -*- coding: utf-8 -*-
class HTMLOutputer(object):

    def __init__(self):
        self.datas = []

    def collect_data(self, new_data):
        if new_data is None:
            return
        else:
            self.datas.append(new_data)

    def output_html(self):
        fout = open('output.html', 'w')
        fout.write("<html>")
        fout.write("<body>")
        fout.write("<table>")
        for data in self.datas:
            fout.write("<tr>")
            # fout.write("<td>%s</td>" % data['url'])
            # fout.write("<td>%s</td>" % data['title'].encode('utf-8'))
            # fout.write("<td>%s</td>" % data['summary'].encode('utf-8'))
            fout.write("<td>")
            fout.write("<a href=\"")
            fout.write(data['url'])
            fout.write("\">")
            fout.write("%s" % data['title'].encode('utf-8').decode('utf-8'))
            fout.write("</a>")
            fout.write("</td>")
            fout.write("<td>")
            try:
                fout.write("%s" % data['summary'].encode('utf-8').decode('utf-8'))
            except:
                fout.write("[not summary found]")
            fout.write("</td>")
            fout.write("</tr>")
        fout.write("</table>")
        fout.write("</body>")
        fout.write("</html>")
        fout.close()

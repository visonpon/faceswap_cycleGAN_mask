import dominate
from dominate.tags import *
import os
import option as opt
import util

class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()
    
    
class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False
    def display_current_results(self, visuals, epoch, save_result):
            if self.display_id > 0:  # show images in the browser
                ncols = self.opt.display_single_pane_ncols
                if ncols > 0:
                    h, w = next(iter(visuals.values())).shape[:2]
                    table_css = """<style>
                            table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                            table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                            </style>""" % (w, h)
                    title = self.name
                    label_html = ''
                    label_html_row = ''
                    nrows = int(np.ceil(len(visuals.items()) / ncols))
                    images = []
                    idx = 0
                    for label, image_numpy in visuals.items():
                        label_html_row += '<td>%s</td>' % label
                        images.append(image_numpy.transpose([2, 0, 1]))
                        idx += 1
                        if idx % ncols == 0:
                            label_html += '<tr>%s</tr>' % label_html_row
                            label_html_row = ''
                    white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                    while idx % ncols != 0:
                        images.append(white_image)
                        label_html_row += '<td></td>'
                        idx += 1
                    if label_html_row != '':
                        label_html += '<tr>%s</tr>' % label_html_row
                    # pane col = image row
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                else:
                    idx = 1
                    for label, image_numpy in visuals.items():
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1

            if self.use_html and (save_result or not self.saved):  # save images to a html file
                self.saved = True
                for label, image_numpy in visuals.items():
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)
                # update website
                webpage = HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
                for n in range(epoch, 0, -1):
                    webpage.add_header('epoch [%d]' % n)
                    ims = []
                    txts = []
                    links = []

                    for label, image_numpy in visuals.items():
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                    webpage.add_images(ims, txts, links, width=self.win_size)
                webpage.save()
